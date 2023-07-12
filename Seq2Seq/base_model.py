import torch
from torch import nn
from config import Config
from torch.autograd import Variable
import torch
import math

config = Config()


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        seq_len, batch_size, emb_dim = x.size()
        outputs = self.relu(self.layer(x.contiguous().view(-1, emb_dim)))
        outputs = outputs.view(seq_len, batch_size, -1)
        dec_hidden = torch.sum(outputs, 0)  # 把所有seq加起来表示总的输出
        return outputs, dec_hidden.unsqueeze(0)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim, encoder_hidden_size):
        super(Decoder, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, bidirectional=False)

        self.attention_module = Attention(encoder_hidden_size, hidden_size)
        self.W_combine = nn.Linear(embedding_dim + encoder_hidden_size, hidden_size)
        self.W_out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, prev_y_batch, prev_h_batch, encoder_outputs_batch):
        # 上一时刻隐藏层输出q 与 编码器输出k 计算注意力相关系数alpha，输出大小 b * seq_len
        attention_weights = self.attention_module(prev_h_batch, encoder_outputs_batch)
        # 根据alpha对编码器输出进行加权，获得v
        # attention_weights.unsqueeze(1) -> b * 1 * seq_len
        # encoder_outputs_batch.transpose(0, 1) -> b * seq_len * encoder_output_dim
        # context = bmm((b,1,seq_len),(b,seq_len,encoder_output_dim)) -> b * 1 * encoder_output_dim
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


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim=None):
        super(Attention, self).__init__()
        self.num_directions = 1
        self.h_dim = encoder_dim
        self.s_dim = decoder_dim
        self.a_dim = self.s_dim if attention_dim is None else attention_dim

        self.U = nn.Linear(self.h_dim * self.num_directions, self.a_dim)
        self.W = nn.Linear(self.s_dim, self.a_dim)
        self.v = nn.Linear(self.a_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, prev_h_batch, encoder_outputs):
        src_seq_len, batch_size, enc_dim = encoder_outputs.size()
        # uh -> seq_len * b * a_dim
        uh = self.U(encoder_outputs.view(-1, self.h_dim)).view(src_seq_len, batch_size, self.a_dim)
        # wq -> 1 * b * a_dim
        wq = self.W(prev_h_batch.view(-1, self.s_dim)).unsqueeze(0)
        wq3d = wq.expand_as(uh)
        wquh = self.tanh(wq3d + uh)
        attention_scores = self.v(wquh.view(-1, self.a_dim)).view(batch_size, src_seq_len)
        attention_weights = self.softmax(attention_scores)
        return attention_weights



