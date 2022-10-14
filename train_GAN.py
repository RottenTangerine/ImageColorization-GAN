from MyDataset import ColorizeData
from GAN_model import Discriminator, Generator
from loss_GAN import LossGAN

from torch.utils.data import DataLoader, random_split
import torch
import time
import os

# model_number
time = time.gmtime()[:6]
model_id = ''
for i in time:
    model_id += str(i)

# hyper params
batch_size = 8
num_epochs = 80
G_learning_rate = 1e-4
D_learning_rate = 1e-5


# dataset
dataset = ColorizeData('dataset/colored_manga/color_full')
split_ratio = 0.2
train_dataset, validate_dataset = random_split(dataset, [l:=round(len(dataset) * (1 - split_ratio)), len(dataset) - l])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validate_loader = DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=True)

print(len(train_dataset), len(validate_dataset))


# initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = Generator().to(device)
D = Discriminator().to(device)

criterion = LossGAN(num_epochs, cuda=True)

optimizer_G = torch.optim.Adam(G.parameters(), lr=G_learning_rate)
optimizer_D = torch.optim.Adam(D.parameters(), lr=D_learning_rate)

scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=1, gamma=0.2)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=1, gamma=0.1)

# retrain

checkpoint = torch.load('checkpoint/2022101119128_0.pt')
G.load_state_dict(checkpoint['G_state_dict'])
D.load_state_dict(checkpoint['D_state_dict'])

# train
for epoch in range(num_epochs):
    G_loss = 0
    D_loss = 0
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
            print(f'Training Epoch: {epoch + 1}/{num_epochs} [{i}/{len(train_loader)}]\tG_Loss: {G_loss.item():0.4f}\tD_Loss: {D_loss.item():0.4f}\t'
                  f'LR: G:{optimizer_G.state_dict()["param_groups"][0]["lr"]:0.8f}, '
                  f'D:{optimizer_D.state_dict()["param_groups"][0]["lr"]:0.8f}')
    scheduler_G.step()
    scheduler_D.step()

    # check point
    os.makedirs('checkpoint', exist_ok=True)
    torch.save({'epoch': epoch,
                'G_state_dict':G.state_dict(),
                'D_state_dict':D.state_dict(),
                'G_loss': G_loss,
                'D_loss': D_loss,
                }, f'checkpoint/{model_id}_{epoch}.pt')

# validate


# save model
os.makedirs('trained_model', exist_ok=True)
torch.save(G.state_dict(), f'./trained_model/{model_id}_GAN_G.pth')
torch.save(D.state_dict(), f'./trained_model/{model_id}_GAN_D.pth')