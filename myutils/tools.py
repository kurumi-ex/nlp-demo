import numpy as np
import torch
import matplotlib.pyplot as plt
from ch8.timemachine.timemachine_process import Vocabulary


def draw_train_pic(y, pic_name: str):
    y = torch.tensor(y).numpy()
    x = np.arange(len(y)) + 1
    plt.plot(x, y)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.title(pic_name)
    plt.show()


def predict_ch8(prefix, vocab: Vocabulary, num_pred, net):
    """
    :param prefix: [1, 0, 11]
    :param vocab: ['a', 'b', 'c', 'd']
    :param num_pred: 10
    :param net: torch.nn.Module
    :return:
    """
    res = []
    state = None
    for x in prefix:
        _, state = net(torch.tensor(x).reshape(1, 1), state)
        res.append(x)
    for _ in range(num_pred):
        y, state = net(torch.tensor(res[-1]).reshape(1, 1), state)
        res.append(int(y.argmax(dim=-1).reshape(1)))
    return "".join([vocab.get_token(x) + ' ' for x in res])


def get_mash():
    pass