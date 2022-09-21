import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import warnings
warnings.filterwarnings('ignore')


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):  # 如果还需要其他参数的话，就用**kwargs返回参数的名称和值
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class PositionalEncoding(nn.Module):
    "位置编码"
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # 将它放入到模型当中，但不会更新模型参数，相当于是模型的常数

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return x


class AddNorm(nn.Module):
    """残差连接后进行层规范化 不改变数据维度格式"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


def subsequent_mask(size):
    " sequence mask"
    attn_shape = (1, size, size)
    # 返回一个三角矩阵，因为只是标记，所以使用uint8类型的数据，比较简单
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0 # 每一个数据都是bool类型的下三角矩阵


def attention(query, key, value, mask=None, dropout=None):
    d_k = math.sqrt(query.size(-1))  # d_k小数也可以
    scores = torch.matmul(query, key.transpose(-2, -1)) / d_k
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = nn.functional.softmax(scores, dim=-1)
    if dropout is not None:  # 选择是否使用dropout
        p_attn=dropout(p_attn)
    return torch.matmul(p_attn, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout_p=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_head == 0
        self.d_k = d_model // num_head
        self.num_head = num_head
        self.W_q = nn.Linear(d_model, d_model)  # 一定要同时创建多个线形层，这样才不一样
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x_q, x_k, x_v, mask):
        batch_size = x_q.size(0)
        query = self.W_q(x_q).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)  # 先整体计算曾以权重wq，再分头，分头之后各自计算注意力
        key = self.W_k(x_k).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)
        value = self.W_v(x_v).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)  # 增加一个维度，且应用到所有的上面
        attn_out = attention(query, key, value,mask=mask, dropout=self.dropout).permute(0, 2, 1, 3).contiguous()
        concat_out = attn_out.view(batch_size, x_q.size(1), -1)  # concat attention
        del query
        del key
        del value  # 清除内存
        return self.W_o(concat_out)


class Encoderlayer(nn.Module):
    def __init__(self, d_model, norm_size, fnn_size, num_heads, dropout, **kwargs):
        super(Encoderlayer, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(d_model=d_model, num_head=num_heads, dropout_p=dropout)
        self.addnorm1 = AddNorm(normalized_shape=norm_size, dropout=dropout)
        self.fnn = PositionWiseFFN(ffn_num_input=d_model, ffn_num_hiddens=fnn_size, ffn_num_outputs=d_model)
        self.addnorm2 = AddNorm(normalized_shape=norm_size, dropout=dropout)

    def forward(self, x, mask):
        add_1_out = self.addnorm1(x, self.attention(x, x, x, mask))
        return self.addnorm2(add_1_out, self.fnn(add_1_out))


class encoder(nn.Module):
    def __init__(self,vocab_size, d_model, norm_size, fnn_size, num_heads, dropout_p, num_layers,  **kwargs):
        super(encoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.postioncoding = PositionalEncoding(d_model=d_model, dropout=dropout_p)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module('encoder_block' + str(i),
                                 Encoderlayer(d_model=d_model, norm_size=norm_size, fnn_size=fnn_size,
                                              num_heads=num_heads, dropout=dropout_p))  # 增加模块的代码

    def forward(self, x,mask, *args):
        x = self.postioncoding(self.embedding(x) * math.sqrt(self.d_model))
        for i ,layer in enumerate(self.blks):
            x = layer(x, mask)
        return x


class decoderlayer(nn.Module):
    def __init__(self, d_model, norm_size, fnn_size, num_heads, dropout, **kwargs):
        super(decoderlayer, self).__init__()
        self.attention1 = MultiHeadAttention(d_model=d_model, num_head=num_heads, dropout_p=dropout)
        self.addnorm1 = AddNorm(normalized_shape=norm_size, dropout=dropout)
        self.attention2 = MultiHeadAttention(d_model=d_model, num_head=num_heads, dropout_p=dropout)
        self.addnorm2 = AddNorm(normalized_shape=norm_size, dropout=dropout)
        self.ffn = PositionWiseFFN(ffn_num_input=d_model, ffn_num_hiddens=fnn_size, ffn_num_outputs=d_model)
        self.addnorm3 = AddNorm(normalized_shape=norm_size, dropout=dropout)

    # 2个注意力计算，一个前馈神经网络
    def forward(self, x, kv_memory, src_mask=None, tgt_amsk=None):
        """
        训练阶段，输出序列的所有词元都在同一时间处理，
        预测阶段，输出序列是通过词元一个接着一个解码，
        """
        m = kv_memory
        x2 = self.attention1(x, x, x, tgt_amsk)
        x2 = self.addnorm1(x, x2)
        # 编码器－解码器注意力。
        x3 = self.attention2(x2, m, m, src_mask)
        Z = self.addnorm2(x3, x3)
        return self.addnorm3(Z, self.ffn(Z))


class decoder(d2l.AttentionDecoder):
    def __init__(self,vocab_size, d_model, norm_size, fnn_size, num_heads, dropout_p, num_layers,  **kwargs):
        super(decoder, self).__init__()
        self.d_model = d_model
        self.num_layer = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model, dropout=0.1)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module('decoder_unit'+str(i),
                                 decoderlayer(
                                    d_model=d_model, norm_size=norm_size,
                                     fnn_size=fnn_size, num_heads=num_heads, dropout=dropout_p,

                                 ))
        self.dense = nn.Linear(d_model, vocab_size)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x = self.pos_encoding(self.embedding(x)* math.sqrt(self.d_model))
        for i, layer in enumerate(self.blks):
            x = layer(x, memory, src_mask, tgt_mask)
        return self.dense(x)


decoder_ = decoder(vocab_size=28, d_model=256, norm_size=[40, 256], fnn_size=154, num_heads=8, dropout_p=0.2, num_layers=1)
test = torch.randint(low=0, high=28, size=(32, 40))
memory = torch.randn((32, 40, 256))
out =decoder_(test, memory)
print(out.shape)

