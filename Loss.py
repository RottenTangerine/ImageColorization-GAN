import torch
import torch.nn as nn
import torchvision.models as models

class BasicLoss(nn.Module):
    def __init__(self, cuda=False):
        super(BasicLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.l1_loss = nn.L1Loss()

        if cuda:
            self.mse = self.mse.cuda()
            self.smooth_l1_loss = self.smooth_l1_loss.cuda()
            self.l1_loss = self.l1_loss.cuda()


        self.resnet = models.resnet34(pretrained=True).eval()
        if cuda:
            self.resnet = self.resnet.cuda()
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, pred, target, epoch_id):
        if epoch_id % 2 == 0:
            pixel_loss = self.l1_loss(pred, target)
        else:
            pixel_loss = self.mse(pred, target)

        res_target = self.resnet(target).detach()
        res_pred = self.resnet(pred)
        perceptual_loss = self.mse(res_pred, res_target)
        return pixel_loss, perceptual_loss




