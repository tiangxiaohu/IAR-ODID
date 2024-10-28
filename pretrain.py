from util import *
from model import *
from dataloader import *
from sklearn.decomposition import PCA

class PretrainModelManager:

    def __init__(self, args, data):
        set_seed(args.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.bert_model == "sentence-transformers/paraphrase-mpnet-base-v2":
            self.model = MPNetForModel(args.bert_model, num_labels=data.n_known_cls)
        else:
            self.model = BertForModel.from_pretrained(args.bert_model, cache_dir="", num_labels=data.n_known_cls)

        '''
        b = torch.load("./MTP2step")
        for key in list(b.keys()):
            if "backbone" in key:
                b[key[9:]] = b.pop(key)
        self.model.load_state_dict(b, strict=False)
        '''
        # 在预训练过程中只更新最后一层或者pooler的参数
        if args.freeze_bert_parameters_pretrain:
            self.freeze_parameters(self.model)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.data = data
        self.num_train_optimization_steps = int(
            len(data.train_labeled_examples) / args.train_batch_size) * args.num_pretrain_epochs
        self.optimizer = self.get_optimizer(args)
        self.best_eval_score = 0

    def eval(self, args):
        self.model.eval()
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, self.data.n_known_cls)).to(self.device)
        for batch in tqdm(self.data.eval_dataloader, desc="Eval Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                _, logits = self.model(input_ids, segment_ids, input_mask, mode='eval')
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))
        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        return acc

    def train1(self, args):
        self.model.to(self.device)
        print("Start finetune in labeled dataset")
        self.best_eval_score = 0
        wait = 0
        best_model = None
        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):

            self.model.train()

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(self.data.train_labeled_dataloader, desc="Train Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):

                    # 对比学习
                    # 初始化标签矩阵，并根据标签填充
                    label_matrix = torch.zeros(input_ids.size(0), input_ids.size(0))
                    labels = label_ids
                    for i in range(input_ids.size(0)):
                        label_matrix[i] = (labels == labels[i])  # 32*32 每行代表一个样本，每列代表一个类别
                    feats = self.model(input_ids, segment_ids, input_mask, mode="sim", feature_ext=True)  # 32*768
                    feats = F.normalize(feats, 2)  # 32*768
                    # 计算相似度矩阵，并去除对角线元素
                    sim_matrix = torch.exp(torch.matmul(feats, feats.t()) / args.t)  # 计算相似度 32*32
                    sim_matrix = sim_matrix - sim_matrix.diag().diag()  # 去除对角线元素  除去自己和自己的相似度
                    # 获取了正样本，并根据标签矩阵填充
                    pos_matrix = torch.zeros_like(sim_matrix)
                    pos_mask = np.where(label_matrix != 0)  # 找出 label_matrix 中所有非零元素的位置。 里面包含行索引和列索引
                    pos_matrix[pos_mask] = sim_matrix[pos_mask]
                    pos_matrix = torch.topk(sim_matrix, k=1, dim=1).values
                    # 对比学习损失函数
                    cl_loss = pos_matrix / sim_matrix.sum(1).view(-1, 1)
                    cl_loss = cl_loss[cl_loss != 0]
                    cl_loss = -torch.log(cl_loss).mean()
                    if torch.isnan(cl_loss):
                        cl_loss = 0

                    # 交叉熵
                    loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode="train")
                    loss = loss + cl_loss
                    loss.backward()
                    tr_loss += loss.item()

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print('train_loss', loss)

            eval_score = self.eval(args)
            print('eval_score', eval_score)

            if eval_score > self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break

        self.model = best_model
        if args.save_model:
            self.save_model(args)

    def train(self, args):
        self.model.to(self.device)
        print("Start finetune in labeled dataset")
        self.best_eval_score = 0
        wait = 0
        best_model = None
        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(self.data.train_labeled_dataloader, desc="Train Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                with torch.set_grad_enabled(True):
                    loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode="train")
                    loss.backward()
                    tr_loss += loss.item()

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps
            print('train_loss', loss)

            eval_score = self.eval(args)
            print('eval_score', eval_score)

            if eval_score > self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break

        self.model = best_model
        if args.save_model:
            self.save_model(args)
    def get_features_labels(self, dataloader, model, args):

        model.eval()
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.no_grad():
                feature = model(input_ids, segment_ids, input_mask, feature_ext=True)

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels

    def cluster_based(self, representations, n_cluster: int, n_pc: int):
        """
        基于聚类的方法改善输入表示的各向同性。

        参数:
        representations: 输入的表示数组，形状为(n_samples, n_dimensions)。
        n_cluster: 聚类的数量。
        n_pc: 要丢弃的方向的数量。

        返回:
        各向同性的表示数组，形状为(n_samples, n_dimensions)。
        """
        # 使用k-means算法进行聚类，计算质心和分配标签
        km = KMeans(n_cluster).fit(representations)
        label = km.labels_

        # 计算每个聚类的均值表示
        cluster_mean = []
        for i in range(max(label) + 1):
            sum = np.zeros([1, 768])
            for j in np.nonzero(label == i)[0]:
                sum = np.add(sum, representations[j])
            cluster_mean.append(sum / len(label[label == i]))

        # 将每个表示减去所属聚类的均值，得到零均值表示
        zero_mean_representation = []
        for i in range(len(representations)):
            zero_mean_representation.append((representations[i]) - cluster_mean[label[i]])

        # 按聚类整理零均值表示
        cluster_representations = {}
        for i in range(n_cluster):
            cluster_representations.update({i: {}})
            for j in range(len(representations)):
                if (label[j] == i):
                    cluster_representations[i].update({j: zero_mean_representation[j]})

        # 将聚类表示转换为数组形式，方便后续处理
        cluster_representations2 = []
        for j in range(n_cluster):
            cluster_representations2.append([])
            for key, value in cluster_representations[j].items():
                cluster_representations2[j].append(value)

        cluster_representations2 = np.array(cluster_representations2)

        # 使用主成分分析（PCA）对每个聚类的表示进行降维
        model = PCA()
        post_rep = np.zeros((representations.shape[0], representations.shape[1]))

        for i in range(n_cluster):
            model.fit(np.array(cluster_representations2[i]).reshape((-1, 768)))
            component = np.reshape(model.components_, (-1, 768))

            for index in cluster_representations[i]:
                sum_vec = np.zeros((1, 768))

                for j in range(n_pc):
                    sum_vec = sum_vec + np.dot(cluster_representations[i][index],
                                               np.transpose(component)[:, j].reshape((768, 1))) * component[j]

                post_rep[index] = cluster_representations[i][index] - sum_vec

        # 清除输出，准备返回结果
        clear_output()

        return post_rep
    def get_optimizer(self, args):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.lr_pre,
                             warmup=args.warmup_proportion,
                             t_total=self.num_train_optimization_steps)
        return optimizer

    # def save_model(self, args, pre=False):
    #     if not os.path.exists(args.pretrain_dir):
    #         os.makedirs(args.pretrain_dir)
    #         self.save_model = self.model.module if hasattr(self.model, 'module') else self.model
    #     model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
    #     model_config_file = os.path.join(args.pretrain_dir, CONFIG_NAME)
    #     torch.save(self.save_model.state_dict(), model_file)
    #     with open(model_config_file, "w") as f:
    #         f.write(self.save_model.config.to_json_string())

    def save_model(self, args, pre=False):
        if not os.path.exists(args.pretrain_dir):
            os.makedirs(args.pretrain_dir)
        # 确保 self.model 是模型的实例，并且检查是否需要访问 .module
        self.save_model = self.model.module if hasattr(self.model, 'module') else self.model
        model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model_config_file = os.path.join(args.pretrain_dir, CONFIG_NAME)
        # 保存模型状态字典和配置文件
        torch.save(self.save_model.state_dict(), model_file)
        with open(model_config_file, "w") as f:
            f.write(self.save_model.config.to_json_string())
    def freeze_parameters(self, model):
        for name, param in model.bert.named_parameters():
            param.requires_grad = False
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True
