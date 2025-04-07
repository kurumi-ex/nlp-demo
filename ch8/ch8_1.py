import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

from mymodel import *
from mydata import *

T = 1000
w = 0.01
time = torch.linspace(0, T, T)
y = torch.cos(time * w) + torch.normal(torch.zeros(T), torch.ones(T) - 0.8)
print(y.shape)
plt.plot(time, y)
plt.show()

model = SimpleFC()
train_iter = DataLoader(TimeDataset(y), shuffle=True, batch_size=64)
loss = torch.nn.MSELoss()
epochs = 100
lr = 0.0001

trainer = torch.optim.Adam(model.parameters(), lr)
for epoch in range(epochs):
    cur_loss = 0
    for x, label in train_iter:
        trainer.zero_grad()
        l = loss(model(x), label)
        cur_loss += l.item()
        l.backward()
        trainer.step()
    print(f'epoch {epoch + 1}, '
          f'loss: {cur_loss / len(train_iter)}')

model.eval()
with torch.no_grad():
    pred_T = T - 6
    pred_x = y.unfold(dimension=0, size=6, step=1)
    pred_y = model(pred_x)

pred_y = pred_y.reshape(995)
plt.plot(torch.linspace(6, 1000, 995), pred_y)
plt.show()
