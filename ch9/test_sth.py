import torch
from torch import nn

from myutils.tools import predict_seq2seq
from s2s_model import SimpleSeq2seq
from translation_data_process import get_data

s1 = "thanks ."
src_vocab, tar_vocab = get_data(voc=True)
prefix = [src_vocab.get_index(e) for e in s1.split(" ")]
model = SimpleSeq2seq(len(src_vocab), len(tar_vocab), 256, 0.1, hidden_size=256)
model.load_state_dict(torch.load("./runs/seq2seq_best.pt"))
print(predict_seq2seq(prefix, tar_vocab, model))
