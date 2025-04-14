import torch
from torch.utils.data import DataLoader, Dataset
import os

from mymodel import *
from mydata import *
from myutils.tools import draw_train_pic
from timemachine.timemachine_process import get_vocabulary

if not os.path.exists("runs"):
    os.makedirs("runs")

vocab, corpus = get_vocabulary()
dataset = TimeMachineDataset(corpus, seq_len=8)
loader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False)

vocab_size = len(vocab)
hidden_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleRNN(vocab_size, hidden_size)
model.to(device)
epochs = 50
lr = 1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

loss_history = []
best_loss = 0x3f3f3f
for epoch in range(epochs):
    cur_loss = 0
    for x, y in loader:
        optimizer.zero_grad()
        y_pred, _ = model(x.to(device))
        loss = loss_fn(y_pred.view(-1, vocab_size), y.view(-1).to(device))
        loss.backward()
        optimizer.step()
        cur_loss += loss.item()
    loss_history.append(cur_loss/len(loader))
    print(f"{epoch + 1} loss: {cur_loss / len(loader)}")
    if best_loss > cur_loss:
        best_loss = cur_loss
        torch.save(model.state_dict(), "./runs/best_simple_rnn.pth")

torch.save(model.state_dict(), "./runs/last_simple_rnn.pth")
draw_train_pic(loss_history, "simple_rnn loss")
