import math

from torch import nn
import torch
import torch.nn.functional as F


def _get_dot_mat(x, head_num):
    b, seq_len, hidden_num = x.size()
    x = x.reshape(b, seq_len, head_num, hidden_num // head_num)
    return x.transpose(1, 2)


class FFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class NormAndAdd(nn.Module):
    def __init__(self, embedding_dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embedding_dim)

    def forward(self, x, y):
        # 残差相加 单句子归一化
        return self.ln(x + self.dropout(y))


class MultiAttention(nn.Module):
    def __init__(self, head_num, query_size, key_size, value_size, vec_size, output_size,
                 bias=False, dropout=0.1):
        """
        vec_size : 一个做注意力的容量
        """
        super().__init__()
        self.head_num = head_num
        self.hidden_num = head_num * vec_size
        # 直接将多头拼起来
        self.wq = nn.Linear(query_size, self.hidden_num, bias=bias)
        self.wk = nn.Linear(key_size, self.hidden_num, bias=bias)
        self.wv = nn.Linear(value_size, self.hidden_num, bias=bias)
        self.wo = nn.Linear(self.hidden_num, output_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask: bool = None):
        # b seq_len head_num * vec_size ---> b seq_len head_num vec_size
        # b seq_len head_num vec_size ---> b head_num seq_len vec_size
        queries = _get_dot_mat(self.wq(q), self.head_num)
        keys = _get_dot_mat(self.wk(k), self.head_num)
        values = _get_dot_mat(self.wv(v), self.head_num)

        b, _, seq_len, vec_size = queries.size()

        # b head_num seq_len vec_size ---> b head_num vec_size seq_len
        # weights is (b head_num seq_len seq_len)
        weights = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(vec_size)
        # 指定那一维度 则那一维度的元素变化做softmax

        if mask:
            # 生成上三角矩阵 然后填成0
            """
            1 0 0 time1
            1 1 0 time2
            1 1 1 time3
            """
            mask_mat = torch.tril(torch.ones(seq_len, seq_len)).to("cuda")
            weights = weights.masked_fill(mask_mat == 0, float("-inf"))
        softmax_score = F.softmax(weights, dim=-1)
        softmax_score = self.dropout(softmax_score)

        # b head_num seq_len value_size
        z = torch.matmul(softmax_score, values)
        # print(z.shape) torch.Size([16, 9, 5, 8])

        # 多头开始拼接
        # b head_num seq_len value_size ----> b seq_len head_num value_size
        # b seq_len head_num value_size ----> b seq_len head_num * value_size
        z = z.transpose(1, 2)
        z = z.reshape(b, seq_len, self.hidden_num)
        # print(z.shape) torch.Size([16, 5,  head_num * value_size])
        return self.wo(z)


class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, time_steps):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.p = torch.zeros(1, time_steps, embedding_dim, device='cuda')
        idx = torch.arange(time_steps).reshape(time_steps, 1)
        m = torch.pow(10000, torch.arange(0, embedding_dim, 2, dtype=torch.float) / embedding_dim)

        self.p[:, :, 0::2] = torch.sin(idx / m)
        self.p[:, :, 1::2] = torch.cos(idx / m)

    def forward(self, x):
        x = self.embedding(x)
        return x + self.p


class Encoder(nn.Module):
    def __init__(self, hidden_num, vec_size, num_heads, ffn_num_hidden, dropout, use_bias=False, ):
        """
        hidden_num: 输入 和 自注意力出来的向量大小
        num_heads: 注意力数量
        vec_size: 自注意力的时候qkv产生的维度

        """
        super().__init__()
        self.attention = MultiAttention(num_heads, hidden_num, hidden_num, hidden_num, vec_size, hidden_num,
                                        bias=use_bias, dropout=dropout)
        self.res1 = NormAndAdd(hidden_num, dropout)
        self.ffn = FFN(hidden_num, ffn_num_hidden, hidden_num)
        self.res2 = NormAndAdd(hidden_num, dropout)

    def forward(self, x):
        norm_x1 = self.res1(x, self.attention(x, x, x))
        ffn_x = self.ffn(norm_x1)
        return self.res2(norm_x1, ffn_x)


class Decoder(nn.Module):
    def __init__(self, hidden_num, vec_size, num_heads, ffn_num_hidden, dropout, use_bias=False, ):
        """
        hidden_num: 输入 和 自注意力出来的向量大小
        num_heads: 注意力数量
        vec_size: 自注意力的时候qkv产生的维度

        """
        super().__init__()
        self.encoder_y = None
        self.mask_attention = MultiAttention(num_heads, hidden_num, hidden_num, hidden_num, vec_size, hidden_num,
                                             bias=use_bias, dropout=dropout)
        self.res1 = NormAndAdd(hidden_num, dropout)
        self.attention = MultiAttention(num_heads, hidden_num, hidden_num, hidden_num, vec_size, hidden_num,
                                        bias=use_bias, dropout=dropout)
        self.res2 = NormAndAdd(hidden_num, dropout)
        self.ffn = FFN(hidden_num, ffn_num_hidden, hidden_num)
        self.res3 = NormAndAdd(hidden_num, dropout)

    def forward(self, x):
        x1 = self.res1(x, self.mask_attention(x, x, x, mask=True))
        x2 = self.res2(x1, self.attention(self.encoder_y, self.encoder_y, x1))
        return self.res3(x2, self.ffn(x2))

    def set_encoder_y(self, encoder_y):
        self.encoder_y = encoder_y


class TransFormer(nn.Module):
    def __init__(self, src_vocab_size, tar_vocab_size, embedding_dim, time_steps, hidden_num, vec_size, ffn_num_hidden,
                 num_heads, dropout, layer_num, use_bias=False, ):
        """
        hidden_num: 网络中 单个词的维度

        """
        super().__init__()
        self.embedding1 = PositionalEncoding(src_vocab_size, embedding_dim, time_steps)
        self.encoder = nn.Sequential(
            *[Encoder(hidden_num, vec_size, num_heads, ffn_num_hidden, dropout, use_bias) for _ in range(layer_num)]
        )
        self.embedding2 = PositionalEncoding(tar_vocab_size, embedding_dim, time_steps)
        self.decoder = nn.Sequential(
            *[Decoder(hidden_num, vec_size, num_heads, ffn_num_hidden, dropout, use_bias) for _ in range(layer_num)]
        )
        self.fc = nn.Linear(hidden_num, tar_vocab_size)

    def forward(self, x1, x2):
        y1 = self.encoder(self.embedding1(x1))
        for d in self.decoder:
            d.set_encoder_y(y1)
        ans = self.decoder(self.embedding2(x2))
        return self.fc(ans)
