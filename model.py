import torch as torch
from transformers import MPNetTokenizer, MPNetModel
from transformers.models.mpnet import MPNetPreTrainedModel
from util import *
from torch import nn
from transformers import AutoModelForMaskedLM,AutoModel
import math
class BertForModel(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForModel, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)

        # 定义一个全连接层序列，用于对Bert输出进行进一步处理
        self.dense = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob)
        )
        # 定义一个投影层序列，包含两个全连接层和dropout，用于特征学习
        self.proj = nn.Sequential(
            nn.Linear(config.hidden_size, 2 * config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        
         # 定义分类器，用于输出预测结果
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        # 应用权重初始化函数
        self.apply(self.init_bert_weights)


    def mean_pooling(self, model_output, attention_mask):
        """
        使用平均池化方法从模型输出中提取每个序列的表示。
        """
        # 直接使用模型得输出作为token embeddings
        token_embeddings = model_output
        # 将attention_mask扩展到与token embeddings相同的形状，以便进行元素乘法
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # 对token embeddings进行加权平均，权重由attention_mask决定
        # 使用clamp函数来避免除以0的情况，确保数值稳定性
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def loss_covariance(self, embeddings):
        # 计算均值向量
        meanVector = embeddings.mean(dim=0)
        centereVectors = embeddings - meanVector  # 得到中心化后的特征向量

        # 估算协方差矩阵
        featureDim = meanVector.shape[0]  # 特征维度
        dataCount = embeddings.shape[0]  # 数据样本数量
        covMatrix = ((centereVectors.t()) @ centereVectors) / (dataCount - 1)

        # 用相关矩阵
        stdVector = torch.std(embeddings, dim=0)
        sigmaSigmaMatrix = (stdVector.unsqueeze(1)) @ (stdVector.unsqueeze(0))
        normalizedConvMatrix = covMatrix / sigmaSigmaMatrix
        relativeMatrix = normalizedConvMatrix / torch.trace(normalizedConvMatrix)  # 计算相对矩阵
        deltaMatrix = relativeMatrix - torch.eye(featureDim).to("cuda:0")  # 计算与单位矩阵的差异
#         deltaMatrix = relativeMatrix + torch.eye(featureDim).to("cuda:0")  # 计算与单位矩阵的差异
        relLoss = torch.norm(deltaMatrix)  # 计算Frobenius范数

        # # 归一化协方差矩阵
#         stdVector = torch.std(embeddings, dim=0)
#         sigmaSigmaMatrix = (stdVector.unsqueeze(1)) @ (stdVector.unsqueeze(0))
#         normalizedConvMatrix = covMatrix / sigmaSigmaMatrix
#         deltaMatrix = normalizedConvMatrix - torch.eye(featureDim).to("cuda:0")  # 计算与单位矩阵的差异
#         covLoss = torch.norm(deltaMatrix)  # 计算Frobenius范数

        return relLoss

    # 加相关矩阵正则化的
    # def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None, mode=None, centroids=None,
    #             labeled=False, feature_ext=False):
    #
    #     # 使用 BERT 模型获取编码后的第 12 层输出和原始池化输出
    #     encoded_layer_12, pooled_output_ori = self.bert(input_ids, token_type_ids, attention_mask,
    #                                                     output_all_encoded_layers=False)
    #     # 通过平均注意力掩码池化编码层输出，获取更通用的池化输出
    #     pooled_output = self.mean_pooling(encoded_layer_12, attention_mask)
    #
    #     # 对池化输出进行全连接层处理，用于后续分类或投影
    #     pooled_output = self.dense(pooled_output)
    #     proj_output = self.proj(pooled_output)
    #
    #     # 获取分类器的 logits 输出，用于分类或损失计算
    #     logits=self.classifier(pooled_output)
    #     if mode == "sim":
    #         return proj_output
    #     if feature_ext:
    #         return pooled_output
    #     elif mode == 'train':
    #         lossCE = nn.CrossEntropyLoss()(logits, labels)
    #         lossCOV = self.loss_covariance(logits)
    #         loss = 0.9*lossCE + 0.1*lossCOV
    #         return loss
    #     else:
    #         return pooled_output, logits

    def loss_W(self,embedding_output,last_output):
        # 初始化一个与嵌入输出相同类型的最大特征向量集合
        last_feature = torch.zeros(embedding_output.size(0) * embedding_output.size(1),
                                   embedding_output.size(-1)).type_as(embedding_output)

        # 用于跟踪处理的特征数量
        num = 0
        # 遍历嵌入输出的每个位置，计算并归一化最后一个输出特征
        for i in range(embedding_output.size(0)):
            for j in range(embedding_output.size(1)):
                # if attention_mask[i][j]!=0:
                last_feature[num, :] = last_output[i, j, :] / torch.norm(last_output[i, j, :], 2)
                num += 1

        # 计算所有归一化特征的平均值
        mean_feature = torch.mean(last_feature, 0)
        # 计算所有归一化特征的协方差矩阵
        cov_feature = torch.cov(last_feature.T, correction=0)

        # 使用线性代数库计算协方差矩阵的奇异值分解
        u, s, vh = torch.linalg.svd(cov_feature)

        # 检查奇异值是否全部为正，以确保矩阵可逆
        if torch.any(s < 0):
            w_loss = torch.tensor(0)
        else:
            # 根据奇异值分解结果构造半正定的协方差矩阵
            cov_half = torch.mm(torch.mm(u.detach(), torch.diag(torch.pow(s, 0.5))), vh.detach())

            # 构造单位矩阵，用于调整协方差矩阵
            identity = torch.eye(cov_feature.size(0)).type_as(cov_feature)
            # 根据特定的规则调整协方差矩阵
            cov = cov_half - (1 / math.sqrt(1024)) * identity
            # 计算权重损失函数的值
            w_loss = torch.norm(mean_feature, 2) ** 2 + 1 + torch.trace(cov_feature) - 2 / math.sqrt(
                1024) * torch.trace(cov_half)
            # w_loss = torch.norm(mean_feature,2)**2 + torch.norm(cov)**2
            # w_loss = torch.norm(mean_feature,2)**2+1024+torch.trace(cov_feature)-2*torch.trace(cov_half)
            # print('111111111111111111111111111111')
            # print('w_loss'+ str(w_loss.item()))
            # print('222222222222222222222222222222')
            return w_loss

        # w_loss = torch.tensor(0)

    def contrastive(self,X):
        # duplicate input
        batch_size = X['input_ids'].shape[0]
        X_dup = self.duplicateInput(X)

        # get raw embeddings
        batchEmbedding = model.forwardEmbedding(X_dup, beforeBatchNorm=beforeBatchNorm)
        batchEmbedding = batchEmbedding.view((batch_size, 2, batchEmbedding.shape[1]))  # (bs, num_sent, hidden)

        # Separate representation
        z1, z2 = batchEmbedding[:, 0], batchEmbedding[:, 1]

        cos_sim = model.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        logits = cos_sim

        labels = torch.arange(logits.size(0)).long().to(model.device)
        lossVal = model.loss_ce(logits, labels)

        return lossVal
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None, mode=None, centroids=None,
                labeled=False, feature_ext=False):

        # 使用 BERT 模型获取编码后的第 12 层输出和原始池化输出
        # encoded_layer_12, pooled_output_ori = self.bert(input_ids, token_type_ids, attention_mask,output_all_encoded_layers=False)
        outputs = self.bert(input_ids, token_type_ids, attention_mask,output_all_encoded_layers=True)
        encoded_layer_12 = outputs[0][11]  # 最后一层
        pooled_output_ori = outputs[1]
        embedding_output = outputs[0][0] # 第一层
        # 通过平均注意力掩码池化编码层输出，获取更通用的池化输出
        pooled_output = self.mean_pooling(encoded_layer_12, attention_mask)

        # 对池化输出进行全连接层处理，用于后续分类或投影
        pooled_output = self.dense(pooled_output)
        proj_output = self.proj(pooled_output)

        # 获取分类器的 logits 输出，用于分类或损失计算
        logits=self.classifier(pooled_output)
        if mode == "sim":
            return proj_output
        if feature_ext:
            return pooled_output
        elif mode == 'train':
            lossCE = nn.CrossEntropyLoss()(logits, labels)
            loss = lossCE
            return loss
        elif mode == 'train_w':
            lossCE = nn.CrossEntropyLoss()(logits, labels)
            lossW = self.loss_W(embedding_output, encoded_layer_12)
            loss = lossCE + lossW
            return loss
        elif mode == 'train_cl':
            lossCE = nn.CrossEntropyLoss()(logits, labels)
            lossCL = self.contrastive()

        else:
            return pooled_output, logits

class MPNetForModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()

        self.num_labels = num_labels
        self.model_name = model_name
        self.backbone = AutoModel.from_pretrained(model_name)

        self.hidden_size = self.backbone.config.hidden_size
        self.dropout_prob = self.backbone.config.hidden_dropout_prob

        self.dense = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.backbone.config.hidden_dropout_prob)
        )

        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, 2*self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(2*self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
        )
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            labels=None,
            mode=None,
            feature_ext=False
    ):
        if 'bert' in self.model_name:
            outputs = self.backbone(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        else:
            outputs = self.backbone(input_ids, attention_mask)
        pooled_output = self.mean_pooling(outputs, attention_mask)
        pooled_output = self.dense(pooled_output)

        proj_output = self.proj(pooled_output)
        logits = self.classifier(pooled_output)

        if mode == "sim":
            return proj_output
        if feature_ext:
            return pooled_output
        elif mode == "train":
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss
        else:
            return pooled_output, logits