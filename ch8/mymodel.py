from torch import nn


class SimpleFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 15)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(15, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
