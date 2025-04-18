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
        # output的形状:(batch_size,num_steps,num_hidden)
        # state的形状: (batch_size, layers, num_hidden)
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
        # TODO ?
        context = state[-1].repeat(x.shape[0], 1, 1)
        x_and_context = torch.cat((x, context), 2)
        output, state = self.gru(x_and_context, state)
        return self.fc(output), state


class Simple_Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_size, dropout,
                 hidden_size=64, num_layers=2):
        super().__init__()
        self.encoder = Encoder(vocab_size, embedding_size, dropout, )
        self.decoder = Decoder(vocab_size, embedding_size, dropout, )

    def forward(self, src, trg):
        return self.decoder(self.encoder(src))
