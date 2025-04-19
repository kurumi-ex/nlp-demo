import torch
from torch import nn
from torch.utils.data import DataLoader

from translation_data_process import get_data
from translation_data import *
from s2s_model import SimpleSeq2seq
from myutils.tools import draw_train_pic

src_vocab, tar_vocab, src_tensor, tar_tensor = get_data()
train_data = TranslationDataset(src_tensor, tar_tensor)
dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleSeq2seq(len(src_vocab), 32, 0.1)
model.to(device)
model.train()
epochs = 100
lr = 2e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

loss_history = []
min_loss = 0x3f3f3f3f
for epoch in range(epochs):
    for batch in dataloader:
        x, y = batch
        pred, _ = model(x.to(device), y.to(device))
        optimizer.zero_grad()
        print('ok')

draw_train_pic(loss_history, "SimpleSeq2seq train loss")
