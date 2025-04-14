from pathlib import Path
import torch

import myutils.tools as ts
from timemachine.timemachine_process import get_vocabulary
from mymodel import SimpleRNN

vocab, _ = get_vocabulary()
model = SimpleRNN(len(vocab), 256)
model.load_state_dict(torch.load('./runs/best_simple_rnn.pth'))
model.eval()

words = [['i', 'am'], ['the', 'time', 'machine']]

for word in words:
    prefix = [vocab.get_index(w) for w in word]
    print(ts.predict_ch8(prefix, vocab, 2, model))
