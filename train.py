from MyDataset import ColorizeData
from basic_model import Net
from Loss import BasicLoss

from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import time
import os


# dataset
dataset = ColorizeData('landscape_images')
split_ratio = 0.2
train_dataset, validate_dataset = random_split(dataset, [l:=round(len(dataset) * (1 - split_ratio)), len(dataset) - l])
train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
validate_loader = DataLoader(dataset=validate_dataset, batch_size=8, shuffle=True)

print(len(train_dataset), len(validate_dataset))

# hyper params
num_epochs = 30
learning_rate = 1e-3



# initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net().to(device)
# continue training
model.load_state_dict(torch.load('trained_model/basic_model_2022918162044.pth'))

criterion = BasicLoss(cuda=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train
for epoch in range(num_epochs):
    for i, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        out_img = model(data)

        loss_l2, loss_perceptual = criterion(out_img, target, epoch)
        loss = loss_l2 + 0.7 * loss_perceptual

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'Training Epoch: {epoch + 1}/{num_epochs} [{i}/{len(train_loader)}]\tLoss: {loss.item():0.4f}\tLR: {learning_rate:0.4f}')


# validate


# save model
time = time.gmtime()[:6]
s = ''
for i in time:
    s += str(i)
os.makedirs('trained_model', exist_ok=True)
torch.save(model.state_dict(), f'./trained_model/basic_model_{s}.pth')