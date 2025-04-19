import torch
from torch import nn
from s2s_model import Decoder

dc = Decoder(100, 10, 0.1)

x = torch.LongTensor([[1, 2, 3, 4, 5, ], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
state = torch.randn(2, 3, 64)
y,_=dc(x, state)
print(y.size())
