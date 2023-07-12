import torch
import torch.nn as nn
import torch.nn.functional as functions


class TextCNN_Model(nn.Module):

    def __init__(self, configs):
        super(TextCNN_Model, self).__init__()

        len_token = configs.max_sentence_length
        vocab_size = configs.vocab_size
        embedding_dimension = configs.embedding_dimension
        label_num = configs.label_num  # 44类

        filter_sizes = [1, 2, 3, 4]  # 卷积核跨度列表
        n_filters = 1024  # 每类卷积核数量，即将要提取的特征数

        # 词嵌入和dropout
        self.embed = nn.Embedding(vocab_size, embedding_dimension)
        self.position1_embed = nn.Embedding(vocab_size, 10)
        self.position2_embed = nn.Embedding(vocab_size, 10)  # 首尾位置编码
        self.dropout = nn.Dropout(configs.dropout)
        # 卷积列表
        self.conv = nn.ModuleList([nn.Conv2d(1, n_filters,
                                             kernel_size=(fs, embedding_dimension + 20)) for fs in filter_sizes])
        # 线性层用来分类
        self.layer = nn.Linear(len(filter_sizes) * n_filters, label_num)

    def forward(self, sentences, pos1, pos2):
        word_embed = self.embed(sentences).unsqueeze(dim=1)
        pos1_embed = self.position1_embed(pos1).unsqueeze(dim=1)
        pos2_embed = self.position2_embed(pos2).unsqueeze(dim=1)
        x = torch.cat(tensors=[word_embed, pos1_embed, pos2_embed], dim=-1)
        x = [functions.relu(c(x)).squeeze(3) for c in self.conv]
        x = [functions.max_pool1d(c, c.size(2)).squeeze(2) for c in x]
        x = self.dropout(torch.cat(x, dim=1))
        x = self.layer(x)
        return x
