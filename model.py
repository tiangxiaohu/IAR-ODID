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

        self.dense = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob)
        )

        self.proj = nn.Sequential(
            nn.Linear(config.hidden_size, 2 * config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(2 * config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )

        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    def loss_W(self,embedding_output,last_output):
                # Initialize a set of maximum feature vectors with the same type as the embedding output
        last_feature = torch.zeros(embedding_output.size(0) * embedding_output.size(1),
                                   embedding_output.size(-1)).type_as(embedding_output)

        # Track the number of processed features
        num = 0
        # Iterate over each position in the embedding output, compute and normalize the last output feature
        for i in range(embedding_output.size(0)):
            for j in range(embedding_output.size(1)):
                # if attention_mask[i][j] != 0:
                last_feature[num, :] = last_output[i, j, :] / torch.norm(last_output[i, j, :], 2)
                num += 1

        # Compute the mean of all normalized features
        mean_feature = torch.mean(last_feature, 0)
        # Compute the covariance matrix of all normalized features
        cov_feature = torch.cov(last_feature.T, correction=0)

        # Use the linear algebra library to compute the singular value decomposition of the covariance matrix
        u, s, vh = torch.linalg.svd(cov_feature)

        # Check if all singular values are positive to ensure the matrix is invertible
        if torch.any(s < 0):
            w_loss = torch.tensor(0)
        else:
            # Construct a positive semi-definite covariance matrix based on the SVD results
            cov_half = torch.mm(torch.mm(u.detach(), torch.diag(torch.pow(s, 0.5))), vh.detach())

            # Construct an identity matrix to adjust the covariance matrix
            identity = torch.eye(cov_feature.size(0)).type_as(cov_feature)
            # Adjust the covariance matrix according to specific rules
            cov = cov_half - (1 / math.sqrt(1024)) * identity
            # Compute the value of the weight loss function
            w_loss = torch.norm(mean_feature, 2) ** 2 + 1 + torch.trace(cov_feature) - 2 / math.sqrt(
                1024) * torch.trace(cov_half)
            # w_loss = torch.norm(mean_feature,2)**2 + torch.norm(cov)**2
            # w_loss = torch.norm(mean_feature,2)**2+1024+torch.trace(cov_feature)-2*torch.trace(cov_half)
            # print('111111111111111111111111111111')
            # print('w_loss'+ str(w_loss.item()))
            # print('222222222222222222222222222222')
            return w_loss

        # w_loss = torch.tensor(0)
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None, mode=None, centroids=None,
                labeled=False, feature_ext=False):

                # Use the BERT model to get the encoded output of the 12th layer and the original pooled output
        # encoded_layer_12, pooled_output_ori = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        outputs = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        encoded_layer_12 = outputs[0][11]  # Last layer
        pooled_output_ori = outputs[1]
        embedding_output = outputs[0][0]  # First layer
        # Perform mean pooling on the encoded layer output using the attention mask to obtain a more general pooled output
        pooled_output = self.mean_pooling(encoded_layer_12, attention_mask)

        # Process the pooled output through a fully connected layer for subsequent classification or projection
        pooled_output = self.dense(pooled_output)
        proj_output = self.proj(pooled_output)

        # Get the logits output from the classifier for classification or loss calculation
        logits=self.classifier(pooled_output)
        if mode == "sim":
            return proj_output
        if feature_ext:
            return pooled_output
        elif mode == 'train':
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss
        elif mode == 'train_w':
            lossCE = nn.CrossEntropyLoss()(logits, labels)
            lossW = self.loss_W(embedding_output, encoded_layer_12)
            loss = lossCE + lossW
            return loss
        elif mode == "w":
            lossCE = nn.CrossEntropyLoss()(logits, labels)
            lossW = self.loss_W(embedding_output, encoded_layer_12)
            return lossCE,lossW
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
    def loss_W(self, embedding_output, last_output):
        # Initialize a set of maximum feature vectors with the same type as the embedding output
        last_feature = torch.zeros(embedding_output.size(0) * embedding_output.size(1),
                                   embedding_output.size(-1)).type_as(embedding_output)

        # Track the number of processed features
        num = 0
        # Iterate over each position in the embedding output, compute and normalize the last output feature
        for i in range(embedding_output.size(0)):
            for j in range(embedding_output.size(1)):
                # if attention_mask[i][j] != 0:
                last_feature[num, :] = last_output[i, j, :] / torch.norm(last_output[i, j, :], 2)
                num += 1

        # Compute the mean of all normalized features
        mean_feature = torch.mean(last_feature, 0)
        # Compute the covariance matrix of all normalized features
        cov_feature = torch.cov(last_feature.T, correction=0)

        # Use the linear algebra library to compute the singular value decomposition of the covariance matrix
        u, s, vh = torch.linalg.svd(cov_feature)

        # Check if all singular values are positive to ensure the matrix is invertible
        if torch.any(s < 0):
            w_loss = torch.tensor(0)
        else:
            # Construct a positive semi-definite covariance matrix based on the SVD results
            cov_half = torch.mm(torch.mm(u.detach(), torch.diag(torch.pow(s, 0.5))), vh.detach())

            # Construct an identity matrix to adjust the covariance matrix
            identity = torch.eye(cov_feature.size(0)).type_as(cov_feature)
            # Adjust the covariance matrix according to specific rules
            cov = cov_half - (1 / math.sqrt(1024)) * identity
            # Compute the value of the weight loss function
            w_loss = torch.norm(mean_feature, 2) ** 2 + 1 + torch.trace(cov_feature) - 2 / math.sqrt(
                1024) * torch.trace(cov_half)
            # w_loss = torch.norm(mean_feature,2)**2 + torch.norm(cov)**2
            # w_loss = torch.norm(mean_feature,2)**2+1024+torch.trace(cov_feature)-2*torch.trace(cov_half)
            # print('111111111111111111111111111111')
            # print('w_loss'+ str(w_loss.item()))
            # print('222222222222222222222222222222')
            return w_loss

        # w_loss = torch.tensor(0)
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
        elif mode == 'train_w':
            lossCE = nn.CrossEntropyLoss()(logits, labels)
            lossW = self.loss_W(embedding_output, encoded_layer_12)
            loss = lossCE + lossW
            return loss
        else:
            return pooled_output, logits
