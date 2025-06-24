import torch
from torch import nn
from torch.utils.data import DataLoader

from ch9.translation_data_process import get_data
from ch9.translation_data import *
from ch9.s2s_model import MaskedSoftmaxCELoss
from my_transformer import TransFormer
from myutils.tools import draw_train_pic
from ch8.timemachine.timemachine_process import ST

batch_size = 32
src_vocab, tar_vocab, src_tensor, tar_tensor, src_len, tar_len = get_data("../ch9/fra-eng/fra.txt")
train_data = TranslationDataset(src_tensor, tar_tensor, src_len, tar_len)
dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = TransFormer(len(src_vocab), len(tar_vocab), 32, 20, 32,
                    64, 64, 6, 0.1, 1).to(device)
epochs = 200
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = MaskedSoftmaxCELoss()

loss_history = []
min_loss = 0x3f3f3f3f
for epoch in range(epochs):
    cur_loss = 0
    total_tokens = 0
    for x, y, y_valid_len in dataloader:
        optimizer.zero_grad()

        # 获得decoder的输入 在y之前加个开始符号
        sos_token = tar_vocab.get_index(str(ST.SOS))
        sos = torch.LongTensor([sos_token] * y.size(0)).reshape(y.size(0), 1)
        dec_input = torch.cat((sos, y[:, :-1]), dim=1)
        pred = model(x.to(device), dec_input.to(device))

        loss = loss_fn(pred, y.to(device), y_valid_len.to(device))
        loss.backward()
        optimizer.step()

        cur_loss += loss.item()
        total_tokens += y_valid_len.sum().item()
    cur_loss /= total_tokens
    print("epoch %d, loss %.3f" % (epoch + 1, cur_loss))
    loss_history.append(cur_loss)
    if cur_loss < min_loss:
        min_loss = cur_loss
        torch.save(model.state_dict(), "./runs/seq2seq_best.pt")

torch.save(model.state_dict(), "./runs/seq2seq_last.pt")
draw_train_pic(loss_history, "SimpleSeq2seq train loss")
