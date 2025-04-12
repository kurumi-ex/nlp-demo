import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

from mymodel import *
from mydata import *
from timemachine.timemachine_process import get_vocabulary


def predict_ch8(prefix, vocab, num_pred, net):
    """
    :param prefix: [1, 0, 11]
    :param vocab: ['a', 'b', 'c', 'd']
    :param num_pred: 10
    :param net: torch.nn.Module
    :return:
    """
    state = net.begin_state(1)
    res = []
    for x in prefix:
        _, state = net(x.reshape(1,1), state)
        res.append(x)
    for _ in range(num_pred):
        y, state = net(res[-1], state)
        res.append(int(y.argmax(dim=1).reshape(1)))
    return "".join([vocab[x] for x in res])


vocab_size = 10
hidden_size = 512
model = SimpleRNN(vocab_size, hidden_size)

vocab = get_vocabulary()
print(vocab.valid_token_freq)