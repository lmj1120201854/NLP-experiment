import torch
from torch import nn
from config import Config
from torch.autograd import Variable
import torch
import math
from base_model import Encoder, Decoder

config = Config()


class E2EModel(nn.Module):
    def __init__(self, src_vocab_size, target_vocab_size, cfg=config):
        super(E2EModel, self).__init__()
        self.device = cfg.device
        self.cfg = cfg
        self.src_vocab_size = src_vocab_size
        self.target_vocab_size = target_vocab_size
        # 词嵌入层
        self.embedding_mat = nn.Embedding(src_vocab_size, cfg.embedding_dim, padding_idx=self.cfg.PAD_ID)
        self.embedding_dropout_layer = nn.Dropout(self.cfg.embedding_dropout)
        self.encoder = Encoder(input_size=self.cfg.encoder_input_dim,
                               hidden_size=self.cfg.encoder_output_dim)
        self.decoder = Decoder(input_size=self.cfg.decoder_input_dim,
                               hidden_size=self.cfg.decoder_output_dim,
                               output_size=self.target_vocab_size,
                               embedding_dim=self.cfg.embedding_dim,
                               encoder_hidden_size=self.cfg.encoder_output_dim)

    def forward(self, data):
        batch_x, batch_y = data  # source_len * b, target_len * b
        # 词嵌入 seq_len * b * emd_dim
        encoder_input_embedded = self.embedding_mat(batch_x)
        encoder_input_embedded = self.embedding_dropout_layer(encoder_input_embedded)
        # encode
        encoder_outputs, encoder_hidden = self.encoder(encoder_input_embedded)
        # decode
        dec_len = batch_y.size()[0]
        batch_size = batch_y.size()[1]
        dec_hidden = encoder_hidden
        # 解码器的初始输入，全BOS输入
        dec_input = Variable(torch.LongTensor([self.cfg.BOS_ID] * batch_size)).to(self.device)
        # 解码器输出概率，初始化为全零
        logits = Variable(torch.zeros(dec_len, batch_size, self.target_vocab_size)).to(self.device)

        # teacher forcing
        for di in range(dec_len):
            # 上一输出的词嵌入 b * emb_dim
            prev_y = self.embedding_mat(dec_input)
            # 解码
            dec_output, dec_hidden, attention_weights = self.decoder(prev_y, dec_hidden, encoder_outputs)
            logits[di] = dec_output  # 记录输出词概率
            dec_input = batch_y[di]  # teacher forcing，下一词为标准答案

        return logits  # size: dec_len * batch_size * target_vocab_size

    def predict(self, input_var):
        # embedding
        encoder_input_embedded = self.embedding_mat(input_var)
        # encode
        encoder_outputs, encoder_hidden = self.encoder(encoder_input_embedded)
        # decode
        dec_ids, attention_w = [], []
        curr_token_id = self.cfg.BOS_ID
        curr_dec_idx = 0  # 记录生成的词数
        dec_input_var = Variable(torch.LongTensor([curr_token_id]))
        dec_input_var = dec_input_var.to(self.device)
        dec_hidden = encoder_hidden[:1]  # 1 * b * encoder_dim
        # 生成结束条件为生成EOS或者达到max_target_len
        while curr_token_id != self.cfg.EOS_ID and curr_dec_idx <= self.cfg.max_target_len:
            prev_y = self.embedding_mat(dec_input_var)  # 上一输出的词嵌入,size:b * emb_dim
            # decode
            decoder_output, dec_hidden, decoder_attention = self.decoder(prev_y, dec_hidden, encoder_outputs)
            # 记录注意力alpha
            attention_w.append(decoder_attention.data.cpu().numpy().tolist()[0])
            # 选择最大概率的词作为下一个状态的输入
            topval, topidx = decoder_output.data.topk(1)
            curr_token_id = topidx[0][0]
            # 记录解码结果
            dec_ids.append(int(curr_token_id.cpu().numpy()))
            # 进行下一轮生成
            dec_input_var = (Variable(torch.LongTensor([curr_token_id]))).to(self.device)
            curr_dec_idx += 1
        return dec_ids, attention_w
