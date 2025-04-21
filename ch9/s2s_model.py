import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, dropout,
                 hidden_size=64, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(input_size=embedding_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        output, state = self.gru(x)
        # output的形状:(batch_size, num_steps, num_hidden)
        # state的形状: (layers, batch_size, num_hidden)
        return output, state


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, dropout,
                 hidden_size=64, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(input_size=embedding_size + hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state):
        x = self.embedding(x)
        # 获取 最后一层rnn的最后时刻状态 size为batch_size hidden
        context = state[-1]
        # 将batch_size hidden 这个状态扩展到每个时间段都有一个 res为time_steps batch_size hidden
        context = context.repeat(x.size(1), 1, 1)
        # 调整 batch_size time_steps hidden
        context = context.permute(1, 0, 2)
        x_and_context = torch.cat((x, context), 2)
        output, state = self.gru(x_and_context, state)
        return self.fc(output), state


class SimpleSeq2seq(nn.Module):
    def __init__(self, src_vocab_size, tar_vocab_size, embedding_size, dropout,
                 hidden_size=64, num_layers=2):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embedding_size, dropout, )
        self.decoder = Decoder(tar_vocab_size, embedding_size, dropout, )

    def forward(self, src, tar):
        _, state = self.encoder(src)
        output, state = self.decoder(tar, state)
        return output, state


def sequence_mask(x, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    time_steps = x.size(1)
    # 0-time_steps
    mask = torch.arange(time_steps, dtype=torch.float32, device=x.device)

    # mask成为1 time_steps valid变成batch 1 二者计算会广播
    mask = mask[None, :] < valid_len[:, None]

    # 取反的位置应该置为0
    x[~mask] = value
    return x


class MaskedSoftmaxCELoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        # 设置为 'none' 以便后续手动应用掩码
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none', )

    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        mask = sequence_mask(weights, valid_len)
        unmasked_loss = self.cross_entropy(pred.permute(0, 2, 1), label)
        masked_loss = (unmasked_loss * mask).mean(dim=1)
        return masked_loss.sum()
