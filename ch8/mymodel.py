from torch import nn
import torch


class SimpleFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 15)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(15, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.rnn = nn.RNN(vocab_size, hidden_size, num_layers=2)
        self.fc1 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state=None):
        x = nn.functional.one_hot(x, self.vocab_size).float()
        y, state = self.rnn(x) if state is None else self.rnn(x, state)
        ans = self.fc1(y)
        return ans, state
