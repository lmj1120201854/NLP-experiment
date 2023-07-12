import torch
from torch import nn
from config import Config
from torch.autograd import Variable
import torch
import math
from base_model import Encoder

config = Config()


class EncoderWithSelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderWithSelfAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.self_attention = SelfAttention(self.input_size, self.hidden_size)

    def forward(self, x):
        seq_len, batch_size, emb_dim = x.size()
        x = x.transpose(0, 1)
        outputs = self.self_attention(x)
        outputs = outputs.view(seq_len, batch_size, -1)
        dec_hidden = torch.sum(outputs, 0)  # 把所有seq加起来表示总的输出
        return outputs, dec_hidden.unsqueeze(0)


class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Q = nn.Linear(self.input_dim, self.output_dim)
        self.K = nn.Linear(self.input_dim, self.output_dim)
        self.V = nn.Linear(self.input_dim, self.output_dim)
        self.attn_dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(self.output_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 计算q，k，v
        xq = self.Q(x).unsqueeze(1)
        xk = self.K(x).unsqueeze(1)
        xv = self.V(x).unsqueeze(1)
        # 计算自注意力得分
        attention_score = torch.matmul(xq, xk.transpose(-1, -2))
        attention_score = attention_score / math.sqrt(self.output_dim)
        attention_probs = nn.Softmax(dim=-1)(attention_score)
        attention_probs = self.attn_dropout(attention_probs)
        # 获得具有上下文语义的特征
        context_layer = torch.matmul(attention_probs, xv).squeeze(1)  # bitch_size * seq_len * encoder_output_dim
        context_layer = context_layer.view(-1, self.output_dim)  # -1 * encoder_output_dim
        output = self.relu(self.dense(context_layer))
        return output


class DecoderWithMask(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim, encoder_hidden_size):
        super(DecoderWithMask, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, bidirectional=False)

        self.attention_module = AttentionWithMask(encoder_hidden_size, hidden_size)
        self.W_combine = nn.Linear(embedding_dim + encoder_hidden_size, hidden_size)
        self.W_out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, prev_y_batch, prev_h_batch, encoder_outputs_batch, mask):
        # 上一时刻隐藏层输出q 与 编码器输出k 计算注意力相关系数alpha，输出大小 b * seq_len
        attention_weights = self.attention_module(prev_h_batch, encoder_outputs_batch, mask)
        # 根据alpha对编码器输出进行加权，获得v
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs_batch.transpose(0, 1))
        # 使用单向RNN进行解码，输入张量: b * (prev_y_dim + (enc_dim * num_enc_directions))
        y_ctx = torch.cat((prev_y_batch, context.squeeze(1)), 1)
        rnn_input = self.W_combine(y_ctx)

        dec_rnn_output, dec_hidden = self.rnn(rnn_input.unsqueeze(0), prev_h_batch)
        # 计算概率
        unnormalized_logits = self.W_out(dec_rnn_output[0])
        dec_output = self.log_softmax(unnormalized_logits)
        # 返回最终输出、隐藏状态以及注意力权重
        return dec_output, dec_hidden, attention_weights


class AttentionWithMask(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim=None):
        super(AttentionWithMask, self).__init__()
        self.num_directions = 1
        self.h_dim = encoder_dim
        self.s_dim = decoder_dim
        self.a_dim = self.s_dim if attention_dim is None else attention_dim

        self.U = nn.Linear(self.h_dim * self.num_directions, self.a_dim)
        self.W = nn.Linear(self.s_dim, self.a_dim)
        self.v = nn.Linear(self.a_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, prev_h_batch, encoder_outputs, mask):
        src_seq_len, batch_size, enc_dim = encoder_outputs.size()
        mask = mask if mask <= src_seq_len else src_seq_len
        mask_list = [1] * mask + [0] * (src_seq_len - mask)
        mask_tensor = torch.Tensor(mask_list).unsqueeze(0).to(config.device)
        # uh -> seq_len * b * a_dim
        uh = self.U(encoder_outputs.contiguous().view(-1, self.h_dim)).view(src_seq_len, batch_size, self.a_dim)
        # wq -> 1 * b * a_dim
        wq = self.W(prev_h_batch.view(-1, self.s_dim)).unsqueeze(0)
        wq3d = wq.expand_as(uh)
        wquh = self.tanh(wq3d + uh)
        attention_scores = self.v(wquh.view(-1, self.a_dim)).view(batch_size, src_seq_len)
        attention_scores = torch.multiply(mask_tensor, attention_scores)
        attention_weights = self.softmax(attention_scores)
        return attention_weights


class E2EModelProved(nn.Module):
    def __init__(self, src_vocab_size, target_vocab_size, cfg=config):
        super(E2EModelProved, self).__init__()
        self.device = cfg.device
        self.cfg = cfg
        self.src_vocab_size = src_vocab_size
        self.target_vocab_size = target_vocab_size
        # 词嵌入层
        self.embedding_mat = nn.Embedding(src_vocab_size, cfg.embedding_dim, padding_idx=self.cfg.PAD_ID)
        self.embedding_dropout_layer = nn.Dropout(self.cfg.embedding_dropout)
        self.encoder = Encoder(input_size=self.cfg.encoder_input_dim,
                               hidden_size=self.cfg.encoder_output_dim)
        self.decoder = DecoderWithMask(input_size=self.cfg.decoder_input_dim,
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
            dec_output, dec_hidden, attention_weights = self.decoder(prev_y, dec_hidden, encoder_outputs, di)
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
        mask_count = 1
        # 生成结束条件为生成EOS或者达到max_target_len
        while curr_token_id != self.cfg.EOS_ID and curr_dec_idx <= self.cfg.max_target_len:
            prev_y = self.embedding_mat(dec_input_var)  # 上一输出的词嵌入,size:b * emb_dim
            # decode
            decoder_output, dec_hidden, decoder_attention = self.decoder(prev_y, dec_hidden, encoder_outputs, mask_count)
            mask_count += 1
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



