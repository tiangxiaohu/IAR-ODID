import copy
from collections import Counter
import torch
from model import *
from init_parameter import *
from dataloader import *
from pretrain import *
from util import *
from time import time
import os
torch.set_num_threads(6)
import fitlog
import numpy as np

torch.cuda.empty_cache()
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class ModelManager:
    def __init__(self, args, data, pretrained_model=None):
        set_seed(args.seed)

        if pretrained_model is None:
            pretrained_model = BertForModel.from_pretrained("./model", cache_dir = "", num_labels = data.n_known_cls)
        
        # 如果指定了加载多任务学习的参数，则加载并更新预训练模型的状态
        if args.load_mtp:
            b = torch.load(args.load_mtp)
            # 重命名模型参数链，以适应模型结构
            for key in list(b.keys()):
                if "backbone" in key:
                    b[key[9:]] = b.pop(key)
            pretrained_model.load_state_dict(b, strict=False)

        # 这一步表明和预训练共用一个模型
        # 将预训练模型和设备配置赋值给实例
        self.model = pretrained_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 根据参数决定是否冻结BERT参数
        # 只更新了6-12层参数，且池化层参数选择了更新
        if args.freeze_bert_parameters_em:
            self.freeze_parameters(self.model)
        self.model.to(self.device)

        # 聚类因子才是管初始k值得
        if args.cluster_num_factor > 1:
            self.num_labels = self.predict_k(args, data)
        else:
            self.num_labels = data.num_labels

        # 计算训练步骤总数，用于优化器配置
        num_train_examples = len(data.train_labeled_examples) + len(data.train_unlabeled_examples)
        self.num_train_optimization_steps = int(num_train_examples / args.train_batch_size) * args.num_train_epochs
        # 获取优化器
        self.optimizer = self.get_optimizer(args,self.model)
        # 初始化评价指标和聚类中心等相关变量
        self.best_eval_score = 0
        self.centroids = None
        self.test_results = None
        self.predictions = None
        self.true_labels = None

    def alignment(self, km, args):
        # 关于self.centroids的初始化，在init函数中，初始设置为0
        # 所以第一次运行这个函数时，self.centroids为None
        if self.centroids is not None:

            # 将旧的中心点转移到CPU上，并转换为numpy数组
            old_centroids = self.centroids.cpu().numpy()
            # 获取新的中心点
            new_centroids = km.cluster_centers_

             # 计算旧中心点和新中心点之间的欧几里得距离矩阵
            DistanceMatrix = np.linalg.norm(old_centroids[:, np.newaxis, :] - new_centroids[np.newaxis, :, :], axis=2)
            # 使用线性分配算法找到最佳匹配
            row_ind, col_ind = linear_sum_assignment(DistanceMatrix)

            # 将新的中心点转换为torch张量并转移到适当的设备
            new_centroids = torch.tensor(new_centroids).to(self.device)
            # 初始化新的中心点存储空间
            self.centroids = torch.empty(self.num_labels, args.feat_dim).to(self.device)

            # 根据匹配结果，更新中心点
            alignment_labels = list(col_ind)
            for i in range(self.num_labels):
                label = alignment_labels[i]
                self.centroids[i] = new_centroids[label]

            # 构建从伪标签到新标签的映射
            pseudo2label = {label: i for i, label in enumerate(alignment_labels)}
            # 根据映射更新标签
            pseudo_labels = np.array([pseudo2label[label] for label in km.labels_])

        else: 
            # 如果没有旧的中心点，直接将新的中心点作为旧的中心点，并根据KMeans的标签生成伪标签
            self.centroids = torch.tensor(km.cluster_centers_).to(self.device)
            pseudo_labels = km.labels_

        pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.long).to(self.device)

        return pseudo_labels

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

    def predict_k(self, args, data):

        # 从半监督数据集中提取特征和标签，并将特征转换为numpy数组
        feats, _ = self.get_features_labels(data.train_semi_dataloader, self.model, args)
        feats = feats.cpu().numpy()

        # 使用KMeans算法进行聚类，其中n_clusters设置为数据集中标签的数量
        km = KMeans(n_clusters=data.num_labels).fit(feats)
        y_pred = km.labels_ # 聚类结果的标签分配

        # 获取所有预测标签的唯一值
        pred_label_list = np.unique(y_pred)

        # 计算每个类别至少应包含的样本数量，用于后续判断类别是否被删除
        drop_out = len(feats) / (data.num_labels)
        cnt = 0  # 用于记录将被删除的类别数量

        # 遍历所有预测标签，删除样本数量少于drop_out的类别
        for label in pred_label_list:
            num = len(y_pred[y_pred == label])
            if num < drop_out:
                cnt += 1
        # 输出将被删除的类别数量
        print(cnt)

        # 计算最终保留的类别数量
        num_labels = len(pred_label_list) - cnt
        print('pred_num', num_labels)

        return num_labels

    def get_optimizer(self, args, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.lr,
                             warmup=args.warmup_proportion,
                             t_total=self.num_train_optimization_steps)
        return optimizer

    def evaluation(self, args, data):
        feats, labels = self.get_features_labels(data.test_dataloader, self.model, args)
        feats = feats.cpu().numpy()
        km = KMeans(n_clusters=self.num_labels).fit(feats)
        y_pred = km.labels_
        y_true = labels.cpu().numpy()
        results = clustering_score(y_true, y_pred)
        print('results', results)
        ind, _ = hungray_aligment(y_true, y_pred)
        map_ = {i[0]: i[1] for i in ind}
        y_pred = np.array([map_[idx] for idx in y_pred])
        cm = confusion_matrix(y_true, y_pred)
        print('confusion matrix', cm)
        self.test_results = results
        self.save_results(args)

    def update_pseudo_labels(self, pseudo_labels, args, input_ids,input_mask,segment_ids,label_ids):
        train_data = TensorDataset(input_ids, input_mask, segment_ids, pseudo_labels,
                                   label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        return train_dataloader

    def update_dataset(self,km,feats,k):
        def top_K_idx(data,k):
            data=np.array(data)
            idx=data.argsort()[-k:][::-1]
            return list(idx)
        updata_semi_label=copy.deepcopy(data.semi_label_ids)

        # 遍历训练集中的有标记示例，为每个示例寻找同一类别中的无标记示例
        for a, example in enumerate(data.train_labeled_examples):
            # 获取当前示例的聚类标签
            plabel = km.labels_[a]
            # 初始化同一标签索引列表
            # ndarray[9003]
            same_plabel_idx = []
            # 初始化相似度列表
            top = []
            # 遍历所有聚类标签，寻找与当前示例标签相同的无标记示例索引
            for idx, label in enumerate(km.labels_):
                if label == plabel and idx > len(data.train_labeled_examples):
                    same_plabel_idx.append(idx)
            # 计算当前示例与同一类别中无标记示例的相似度，并存储在top列表中
            for idx in same_plabel_idx:
                top.append(1 - cosine(feats[a], feats[idx]))
            # 选择相似度最高的k个无标记示例索引
            idxlist = top_K_idx(top, k)
            # 从无标记示例索引中选出的top k索引列表
            semi_idxlist = [same_plabel_idx[i] for i in idxlist]
            # 更新这k个无标记示例的半监督标签为当前有标记示例的标签
            for i in semi_idxlist:
                updata_semi_label[i] = updata_semi_label[a]
        return updata_semi_label

    def get_semi_loader(self, semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids, args):
        semi_data = TensorDataset(semi_input_ids, semi_input_mask, semi_segment_ids, semi_label_ids)
        semi_sampler = SequentialSampler(semi_data)
        semi_dataloader = DataLoader(semi_data, sampler=semi_sampler, batch_size = args.train_batch_size)
        return semi_dataloader

    def train(self, args, data):
        bestresults = {'ACC': 0,
                       'ARI': 0,
                       'NMI': 0}
        jsonresults = {}

        for epoch in range(int(args.num_train_epochs)):
            s = time()
            feats, labels = self.get_features_labels(data.train_semi_dataloader, self.model, args)
            feats = feats.cpu().numpy()
            t0 = time()

            # 初始化kmeans模型，进行聚类 self.num_labels来自于超参的聚类因子
            # 如果cluster_num_factor=1，那用直接用数据集里面的类别，否则，与自己用代码预测。
            # 预测代码是predict_k(args, data)
            km = KMeans(n_clusters=self.num_labels).fit(feats)
            t1 = time()
            # 这里记录了kmeans聚类的时间
            kmeans_time = t1 - t0
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            # 将模型设置为训练模式
            self.model.train()
            t0 = time()
            # 通过聚类结果生成伪标签
            pseudo_labels = self.alignment(km, args)

            # 根据是否选择数据增强，更新伪标签和训练集
            if args.augment_data:
                updata_semi_label=self.update_dataset(km,feats,args.k)
                train_semi_dataloader=self.update_pseudo_labels(pseudo_labels, args, data.semi_input_ids,data.semi_input_mask,data.semi_segment_ids,updata_semi_label)
            else:
                train_semi_dataloader = self.update_pseudo_labels(pseudo_labels, args, data.semi_input_ids,data.semi_input_mask,data.semi_segment_ids,data.semi_label_ids)

            for batch in train_semi_dataloader: 
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, pseudo_label_ids, label_ids = batch
                # contrastive loss
                # 初始化标签矩阵，并根据伪标签填充
                label_matrix = torch.zeros(input_ids.size(0), input_ids.size(0))
                labels = pseudo_label_ids
                for i in range(input_ids.size(0)):
                    label_matrix[i] = (labels == labels[i]) # 32*32 每行代表一个样本，每列代表一个类别
                # 获取特征表示，并进行归一化
                feats = self.model(input_ids, segment_ids, input_mask, mode="sim", feature_ext=True) # 32*768
                feats = F.normalize(feats, 2) # 32*768

                # 计算相似度矩阵，并去除对角线元素
                sim_matrix = torch.exp(torch.matmul(feats, feats.t()) / args.t) #计算相似度 32*32
                sim_matrix = sim_matrix - sim_matrix.diag().diag() #去除对角线元素  除去自己和自己的相似度

                # 获取了正样本，并根据标签矩阵填充
                pos_matrix = torch.zeros_like(sim_matrix)
                pos_mask = np.where(label_matrix != 0) # 找出 label_matrix 中所有非零元素的位置。 里面包含行索引和列索引
                pos_matrix[pos_mask] = sim_matrix[pos_mask]
                # 对比学习损失函数
                cl_loss = pos_matrix / sim_matrix.sum(1).view(-1, 1)
                cl_loss = cl_loss[cl_loss != 0]
                cl_loss = -torch.log(cl_loss).mean()
                if torch.isnan(cl_loss):
                    cl_loss = 0

                # cross entropy loss
                ind = (label_ids != -1)
                if any(ind) is False:
                    ce_loss=0
                else:
                    input_ids = input_ids[ind]
                    input_mask = input_mask[ind]
                    segment_ids = segment_ids[ind]
                    label_ids = label_ids[ind]
                    ce_loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode="train")


                loss=(1 - args.beta) * cl_loss + args.beta * ce_loss
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                gradient_accumulation_steps = 16
                if (batch+1) % gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                # self.optimizer.step()
                # self.optimizer.zero_grad()

            t1 = time()
            cont_time = t1 - t0
            tr_loss = tr_loss / nb_tr_steps
            print('train_loss = ', tr_loss)
            e = time()
            print("kmeans:{:.2f} cont:{:.2f} total:{:.2f}".format(kmeans_time, cont_time, e - s))

            # 在测试数据集上进行聚类，评估聚类效果
            feats, labels = self.get_features_labels(data.test_dataloader, self.model, args)
            feats = feats.cpu().numpy()
            km = KMeans(n_clusters=self.num_labels).fit(feats)

            y_pred = km.labels_
            y_true = labels.cpu().numpy()

            # 计算聚类效果的评价指标，并更新最佳结果
            results = clustering_score(y_true, y_pred)
            jsonresults.update({epoch:results})
            print(results)

            if results["ACC"]+results["ARI"]+results["NMI"]>bestresults["ACC"]+bestresults["ARI"]+bestresults["NMI"]:
                bestresults=results

            jsonresults.update({"best": bestresults})

            import json
            info_json=json.dumps(jsonresults,sort_keys=False,indent=4,separators=(",",": "))
            f=open('./outputs/info_{}.json'.format(args.name),'w')
            f.write(info_json)
            torch.cuda.empty_cache()

    # 1加正样本质心;加1个除质心外最相似硬负样本质心
    def train_hardNeg1(self, args, data):
        bestresults = {'ACC': 0,
                       'ARI': 0,
                       'NMI': 0}
        jsonresults = {}

        for epoch in range(int(args.num_train_epochs)):
            s = time()
            feats, labels = self.get_features_labels(data.train_semi_dataloader, self.model, args)
            feats = feats.cpu().numpy()
            t0 = time()

            # 初始化kmeans模型，进行聚类 self.num_labels来自于超参的聚类因子
            # 如果cluster_num_factor=1，那用直接用数据集里面的类别，否则，与自己用代码预测。
            # 预测代码是predict_k(args, data)
            km = KMeans(n_clusters=self.num_labels).fit(feats)
            t1 = time()
            # 这里记录了kmeans聚类的时间
            kmeans_time = t1 - t0
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            # 将模型设置为训练模式
            self.model.train()
            t0 = time()
            # 通过聚类结果生成伪标签
            pseudo_labels = self.alignment(km, args)

            # 根据是否选择数据增强，更新伪标签和训练集
            if args.augment_data:
                updata_semi_label = self.update_dataset(km, feats, args.k)
                train_semi_dataloader = self.update_pseudo_labels(pseudo_labels, args, data.semi_input_ids,
                                                                  data.semi_input_mask, data.semi_segment_ids,
                                                                  updata_semi_label)
            else:
                train_semi_dataloader = self.update_pseudo_labels(pseudo_labels, args, data.semi_input_ids,
                                                                  data.semi_input_mask, data.semi_segment_ids,
                                                                  data.semi_label_ids)

            for batch in train_semi_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, pseudo_label_ids, label_ids = batch

                # contrastive loss

                # 初始化标签矩阵，并根据伪标签填充
                label_matrix = torch.zeros(input_ids.size(0), input_ids.size(0))
                labels = pseudo_label_ids
                for i in range(input_ids.size(0)):
                    label_matrix[i] = (labels == labels[i])  # 32*32 每行代表一个样本，每列代表一个类别

                # 获取特征表示，并进行归一化
                feats = self.model(input_ids, segment_ids, input_mask, mode="sim", feature_ext=True)  # 32*768
                feats = F.normalize(feats, 2)  # 32*768

                centroids = torch.tensor(km.cluster_centers_).to(self.device)
                # 计算样本与所有质心之间的相似度
                similarity_to_centroids = torch.matmul(feats, centroids.t())
                # 对于每个样本 xi，找到最近的质心 ci1 和第二近的质心 ci2

                # 使用 torch.topk 获取每个样本相似度最高的两个质心索引
                topk_indices = torch.topk(similarity_to_centroids, k=2, dim=1).indices
                # 如果你需要，也可以获取相应的相似度值
                topk_values = torch.topk(similarity_to_centroids, k=6, dim=1).values

                # 计算相似度矩阵，并去除对角线元素
                sim_matrix = torch.exp(torch.matmul(feats, feats.t()) / args.t)  # 计算相似度 32*32
                sim_matrix = sim_matrix - sim_matrix.diag().diag()  # 去除对角线元素  除去自己和自己的相似度

                # 获取了正样本，并根据标签矩阵填充
                pos_matrix = torch.zeros_like(sim_matrix)
                pos_mask = np.where(label_matrix != 0)  # 找出 label_matrix 中所有非零元素的位置。 里面包含行索引和列索引
                pos_matrix[pos_mask] = sim_matrix[pos_mask]
                pos_matrix = torch.topk(sim_matrix, k=1, dim=1).values
                # # 取出 topk_values 的第一列
                # topk_values_col1 = topk_values[:, 0].unsqueeze(1)
                # # 将 topk_values 的第一列添加到 pos_matrix 的最后一列
                # pos_matrix = torch.cat([pos_matrix, topk_values_col1], dim=1)

                # # 取出 topk_values 的第二列，即每个样本与次相似质心的相似度
                # topk_values_col2 = topk_values[:, 1:]  # 形状变为 [32, 1]
                # sim_matrix = torch.cat([sim_matrix, topk_values_col2], dim=1)

                # 对比学习损失函数
                cl_loss = pos_matrix / sim_matrix.sum(1).view(-1, 1)
                cl_loss = cl_loss[cl_loss != 0]
                cl_loss = -torch.log(cl_loss).mean()
                if torch.isnan(cl_loss):
                    cl_loss = 0

                # cross entropy loss
                ind = (label_ids != -1)
                if any(ind) is False:
                    ce_loss = 0
                else:
                    input_ids = input_ids[ind]
                    input_mask = input_mask[ind]
                    segment_ids = segment_ids[ind]
                    label_ids = label_ids[ind]
                    ce_loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode="train")

                loss = (1 - args.beta) * cl_loss + args.beta * ce_loss
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                # gradient_accumulation_steps = 16
                # if (batch+1) % gradient_accumulation_steps == 0:
                #     self.optimizer.step()
                #     self.optimizer.zero_grad()
                self.optimizer.step()
                self.optimizer.zero_grad()

            t1 = time()
            cont_time = t1 - t0
            tr_loss = tr_loss / nb_tr_steps
            print('train_loss = ', tr_loss)
            e = time()
            print("kmeans:{:.2f} cont:{:.2f} total:{:.2f}".format(kmeans_time, cont_time, e - s))

            # 在测试数据集上进行聚类，评估聚类效果
            feats, labels = self.get_features_labels(data.test_dataloader, self.model, args)
            feats = feats.cpu().numpy()
            km = KMeans(n_clusters=self.num_labels).fit(feats)

            y_pred = km.labels_
            y_true = labels.cpu().numpy()

            # 计算聚类效果的评价指标，并更新最佳结果
            results = clustering_score(y_true, y_pred)
            jsonresults.update({epoch: results})
            print(results)

            if results["ACC"] + results["ARI"] + results["NMI"] > bestresults["ACC"] + bestresults["ARI"] + bestresults[
                "NMI"]:
                bestresults = results

            jsonresults.update({"best": bestresults})

            import json
            info_json = json.dumps(jsonresults, sort_keys=False, indent=4, separators=(",", ": "))
            f = open('./outputs/info_{}.json'.format(args.name), 'w')
            f.write(info_json)
            torch.cuda.empty_cache()
    # 将其他所有质心加进去--硬负样本
    def train_hardNeg2(self, args, data):
        bestresults = {'ACC': 0,
                       'ARI': 0,
                       'NMI': 0}
        jsonresults = {}

        for epoch in range(int(args.num_train_epochs)):
            s = time()
            feats, labels = self.get_features_labels(data.train_semi_dataloader, self.model, args)
            feats = feats.cpu().numpy()
            t0 = time()

            # 初始化kmeans模型，进行聚类 self.num_labels来自于超参的聚类因子
            # 如果cluster_num_factor=1，那用直接用数据集里面的类别，否则，与自己用代码预测。
            # 预测代码是predict_k(args, data)
            km = KMeans(n_clusters=self.num_labels).fit(feats)
            t1 = time()
            # 这里记录了kmeans聚类的时间
            kmeans_time = t1 - t0
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            # 将模型设置为训练模式
            self.model.train()
            t0 = time()
            # 通过聚类结果生成伪标签
            pseudo_labels = self.alignment(km, args)

            # 根据是否选择数据增强，更新伪标签和训练集
            if args.augment_data:
                updata_semi_label = self.update_dataset(km, feats, args.k)
                train_semi_dataloader = self.update_pseudo_labels(pseudo_labels, args, data.semi_input_ids,
                                                                  data.semi_input_mask, data.semi_segment_ids,
                                                                  updata_semi_label)
            else:
                train_semi_dataloader = self.update_pseudo_labels(pseudo_labels, args, data.semi_input_ids,
                                                                  data.semi_input_mask, data.semi_segment_ids,
                                                                  data.semi_label_ids)

            for batch in train_semi_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, pseudo_label_ids, label_ids = batch

                # contrastive loss

                # 初始化标签矩阵，并根据伪标签填充
                label_matrix = torch.zeros(input_ids.size(0), input_ids.size(0))
                labels = pseudo_label_ids
                for i in range(input_ids.size(0)):
                    label_matrix[i] = (labels == labels[i])  # 32*32 每行代表一个样本，每列代表一个类别

                # 获取特征表示，并进行归一化
                feats = self.model(input_ids, segment_ids, input_mask, mode="sim", feature_ext=True)  # 32*768
                feats = F.normalize(feats, 2)  # 32*768

                # 计算相似度矩阵，并去除对角线元素
                sim_matrix = torch.exp(torch.matmul(feats, feats.t()) / args.t)  # 计算相似度 32*32
                sim_matrix = sim_matrix - sim_matrix.diag().diag()  # 去除对角线元素  除去自己和自己的相似度

                # 获取了正样本，并根据标签矩阵填充
                pos_matrix = torch.zeros_like(sim_matrix)
                pos_mask = np.where(label_matrix != 0)  # 找出 label_matrix 中所有非零元素的位置。 里面包含行索引和列索引
                pos_matrix[pos_mask] = sim_matrix[pos_mask]

                centroids = torch.tensor(km.cluster_centers_).to(self.device)
                # 计算样本与所有质心之间的相似度
                similarity_matrix = torch.matmul(feats, centroids.t())
                # 找到每个样本最相似的质心索引
                max_similarity_indices = similarity_matrix.argmax(dim=1)
                # 获取每个样本与最相似质心的相似度
                max_similarities = similarity_matrix[torch.arange(32), max_similarity_indices]

                # 获取每个样本与其余质心的相似度，形成一个32x76的矩阵
                other_similarities = similarity_matrix.clone()
                other_similarities[torch.arange(32), max_similarity_indices] = 0  # 将最相似的质心相似度设置为0 将最相似的质心相似度设置为0
                other_similarities = other_similarities.to(self.device)
                sim_matrix = torch.cat([sim_matrix, other_similarities], dim=1)

                # 对比学习损失函数
                cl_loss = pos_matrix / sim_matrix.sum(1).view(-1, 1)
                cl_loss = cl_loss[cl_loss != 0]
                cl_loss = -torch.log(cl_loss).mean()
                if torch.isnan(cl_loss):
                    cl_loss = 0

                # cross entropy loss
                ind = (label_ids != -1)
                if any(ind) is False:
                    ce_loss = 0
                else:
                    input_ids = input_ids[ind]
                    input_mask = input_mask[ind]
                    segment_ids = segment_ids[ind]
                    label_ids = label_ids[ind]
                    ce_loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode="train")

                loss = (1 - args.beta) * cl_loss + args.beta * ce_loss
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                # gradient_accumulation_steps = 16
                # if (batch+1) % gradient_accumulation_steps == 0:
                #     self.optimizer.step()
                #     self.optimizer.zero_grad()
                self.optimizer.step()
                self.optimizer.zero_grad()

            t1 = time()
            cont_time = t1 - t0
            tr_loss = tr_loss / nb_tr_steps
            print('train_loss = ', tr_loss)
            e = time()
            print("kmeans:{:.2f} cont:{:.2f} total:{:.2f}".format(kmeans_time, cont_time, e - s))

            # 在测试数据集上进行聚类，评估聚类效果
            feats, labels = self.get_features_labels(data.test_dataloader, self.model, args)
            feats = feats.cpu().numpy()
            km = KMeans(n_clusters=self.num_labels).fit(feats)

            y_pred = km.labels_
            y_true = labels.cpu().numpy()

            # 计算聚类效果的评价指标，并更新最佳结果
            results = clustering_score(y_true, y_pred)
            jsonresults.update({epoch: results})
            print(results)

            if results["ACC"] + results["ARI"] + results["NMI"] > bestresults["ACC"] + bestresults["ARI"] + bestresults[
                "NMI"]:
                bestresults = results

            jsonresults.update({"best": bestresults})

            import json
            info_json = json.dumps(jsonresults, sort_keys=False, indent=4, separators=(",", ": "))
            f = open('./outputs/info_{}.json'.format(args.name), 'w')
            f.write(info_json)
            torch.cuda.empty_cache()
    def eval_pretrain(self):
        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.n_known_cls)).to(self.device)

        for batch in tqdm(data.eval_dataloader, desc="Eval Iteration"):
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
    def load_pretrained_model(self, args):
        pretrained_dict = self.pretrained_model.state_dict()
        classifier_params = ['classifier.weight', 'classifier.bias']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def restore_model(self, args, model):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model.load_state_dict(torch.load(output_model_file))
        return model

    def freeze_parameters(self, model):
        for name, param in model.bert.named_parameters():
            param.requires_grad = False
            param_grad_ = False
            for i in range(6, 12):
                name_str = "encoder.layer." + str(i)
                if name_str in name:
                    param_grad_ = True
            if param_grad_ or "pooler" in name:
                param.requires_grad = True

    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        # var = [args.dataset, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.seed, self.num_labels]
        # names = ['dataset', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor','seed', 'K']
        names = ['dataset', 'alpha', 'beta', 'batch_size', 'seed', 'K',"name"]
        names = ['dataset', 'temp', 'beta', 'batch_size', 'seed', "epoch_pre" , 'epoch' , 'lr_pre' ,'lr', ]
        var = [args.dataset, args.t, args.beta, args.train_batch_size, args.seed, args.num_pretrain_epochs, args.num_train_epochs , args.lr_pre, args.lr]
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(self.test_results, **vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        file_name = '%s.csv' % args.dataset
        results_path = os.path.join(args.save_results_path, file_name)

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results, index=[1])
            df1 = df1.append(new, ignore_index=True)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)

        print('test_results', data_diagram)


if __name__ == '__main__':

    parser = init_model()
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    fitlog.set_rng_seed(args.seed)
    print('_________Prepare {} data__________'.format(args.dataset))
    data = Data(args)

    if args.pretrain:
        manager_p = PretrainModelManager(args, data)
        manager_p.train(args)
        torch.cuda.empty_cache()
        manager = ModelManager(args, data, manager_p.model)
    else:
        args.pretrain_dir = 'pretrained_' + args.dataset
        manager = ModelManager(args, data)

    # print('Training begin...')
    # manager.train(args, data)
    # manager.train_hardNeg1(args, data)
    # manager.train_hardNeg2(args, data)
    print('Training finished!')
    # 加载模型的状态字典
    print('Evaluation begin...')
    manager.evaluation(args, data)

    print('Evaluation finished!')

    # manager.save_results(args)

