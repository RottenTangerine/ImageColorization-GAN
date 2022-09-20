import torch
import torch.nn as nn
import torchvision.models as models

class LossGAN(nn.Module):
    def __init__(self, epochs, cuda=False):
        super(LossGAN, self).__init__()
        self.epochs = epochs
        self.cuda = cuda
        self.mse = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.l1_loss = nn.L1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        if cuda:
            self.mse = self.mse.cuda()
            self.smooth_l1_loss = self.smooth_l1_loss.cuda()
            self.l1_loss = self.l1_loss.cuda()
            self.cross_entropy_loss = self.cross_entropy_loss .cuda()

        self.resnet = models.resnet34(pretrained=True).eval()
        if cuda:
            self.resnet = self.resnet.cuda()
        for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(self, phase, Discriminator, pred, target, epoch_id):
        if phase == 'D':
            # discriminator losses: here we use wan-gp
            # 1. adversarial loss
            adv_D_loss = - torch.mean(Discriminator(target)) + torch.mean(Discriminator(pred.detach()))

            # 2. gradient penalty
            alpha = torch.rand(target.shape[0], 1, 1, 1)
            # eps = self.Tensor(np.random.random((real.size(0), 1, 1, 1)))
            if self.cuda:
                alpha = alpha.cuda(non_blocking=True)
                interpolated_x = (alpha * pred.data + (1.0 - alpha) * target.data).requires_grad_(True)
            else:
                interpolated_x = torch.FloatTensor(
                    alpha * pred.data + (1.0 - alpha) * target.data
                ).requires_grad_(True)
            out = Discriminator(interpolated_x)
            dxdD = torch.autograd.grad(outputs=out,
                                       inputs=interpolated_x,
                                       grad_outputs=torch.ones(out.size()).cuda(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0].view(out.shape[0], -1)
            gp_loss = torch.mean((torch.norm(dxdD, p=2) - 1) ** 2)
            return adv_D_loss, gp_loss, None
        else:
            # generator losses
            # 1. adversarial loss
            adv_G_loss = -torch.mean(Discriminator(pred))

            # 2. pixel loss
            if epoch_id < self.epochs // 4:
                pixel_loss = self.l1_loss(pred, target)
            else:
                pixel_loss = self.mse(pred, target)

            # 3. perceptual loss
            res_target = self.resnet(target).detach()
            res_pred = self.resnet(pred)
            perceptual_loss = self.mse(res_pred, res_target)
            return adv_G_loss, pixel_loss, perceptual_loss