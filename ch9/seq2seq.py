import torch
from torch import nn

embedding = nn.Embedding(100, 4)
print(embedding(torch.ones(16, 10, dtype=torch.long)).shape)
