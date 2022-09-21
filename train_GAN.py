from MyDataset import ColorizeData
from GAN_model import Discriminator, Generator
from loss_GAN import LossGAN

from torch.utils.data import DataLoader, random_split
import torch
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
num_epochs = 80
learning_rate = 2e-4



# initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = Generator().to(device)
D = Discriminator().to(device)

criterion = LossGAN(num_epochs, cuda=True)

optimizer_D = torch.optim.Adam(D.parameters(), lr=learning_rate)
optimizer_G = torch.optim.Adam(G.parameters(), lr=learning_rate)

# train
for epoch in range(num_epochs):
    for i, (img, targets) in enumerate(train_loader):
        img = img.to(device)
        targets = targets.to(device)

        # create noise
        z = torch.rand((len(img), 1, 8, 8)).to(device)

        # Generator
        gen_img = G(img, z)

        # optimize Discriminator
        adv_D_loss, gp_loss, _ = criterion('D', D, gen_img, targets, epoch)
        D_loss = 1.0 * adv_D_loss + 10.0 * gp_loss
        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()

        # optimize Generator
        adv_G_loss, pixel_loss, perceptual_loss = criterion('G', D, gen_img, targets, epoch)
        G_loss = 0.001 * adv_G_loss + 1.0 * pixel_loss + 0.1 * perceptual_loss
        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()



        if i % 100 == 0:
            print(f'Training Epoch: {epoch + 1}/{num_epochs} [{i}/{len(train_loader)}]\tG_Loss: {G_loss.item():0.4f}\tD_Loss: {D_loss.item():0.4f}\tLR: {learning_rate:0.4f}')


# validate


# save model
time = time.gmtime()[:6]
s = ''
for i in time:
    s += str(i)
os.makedirs('trained_model', exist_ok=True)
torch.save(G.state_dict(), f'./trained_model/GAN_G{s}.pth')
torch.save(D.state_dict(), f'./trained_model/GAN_D{s}.pth')