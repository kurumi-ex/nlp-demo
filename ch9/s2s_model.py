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
        x_and_context = torch.cat((x, context), dim=2)
        output, state = self.gru(x_and_context, state)
        return self.fc(output), state


class SimpleSeq2seq(nn.Module):
    def __init__(self, vocab_size, embedding_size, dropout,
                 hidden_size=64, num_layers=2):
        super().__init__()
        self.encoder = Encoder(vocab_size, embedding_size, dropout, )
        self.decoder = Decoder(vocab_size, embedding_size, dropout, )

    def forward(self, src, tar):
        _, state = self.encoder(src)
        output, state = self.decoder(tar, state)
        return output, state
