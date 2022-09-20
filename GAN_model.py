import torch
import torch.nn as nn
from icecream import ic

class CBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
                 stride, padding, weight_init=True, activation='relu', bn=True):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding)
        self.bn = bn
        if bn:
            self.bn = nn.BatchNorm2d(out_channel)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(1e-2)
        else:
            self.activation = nn.Identity()

        if weight_init:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.activation(x)
        return x


class ResBasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
                 stride=1, padding=1, weight_init=True, activation='relu', bn=True):
        super(ResBasicBlock, self).__init__()
        self.conv1 = CBR(in_channel, in_channel, kernel_size,  # in - in
                         stride, padding=padding if padding is not None else (kernel_size - 1)//2,
                         weight_init=weight_init, activation=activation, bn=bn)
        self.conv2 = CBR(in_channel, out_channel, kernel_size,
                         stride, padding=padding if padding is not None else (kernel_size - 1)//2,
                         weight_init=weight_init, activation=activation, bn=bn)
        if stride == 1 and in_channel == out_channel:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = CBR(in_channel, out_channel, kernel_size=1, stride=stride, padding=0,
                                weight_init=False, activation='', bn=False)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(1e-2)
        else:
            self.activation = nn.Identity()


    def forward(self, x):
        i = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = i + out
        out = self.activation(out)
        return out


class EncodeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
                 stride, padding, weight_init=True, activation='relu', bn=True):
        super(EncodeBlock, self).__init__()
        self.conv = CBR(in_channel, out_channel, kernel_size, stride, padding,
                        weight_init=weight_init, activation=activation, bn=bn)
        self.res = ResBasicBlock(out_channel, out_channel, kernel_size, 1, padding,  # out - out
                                 weight_init=weight_init, activation=activation)


    def forward(self, x):
        out = self.conv(x)
        out = self.res(out)
        return out


class Feature(nn.Module):
    def __init__(self, in_channel, out_channel, weight_init=True, activation='leaky', bn=True):
        super(Feature, self).__init__()
        
        self.res1 = ResBasicBlock(in_channel + 1, out_channel, 3,# add noise 
                                  weight_init=weight_init, activation=activation, bn=bn)
        self.res2 = ResBasicBlock(out_channel, out_channel//2, 3,
                                  weight_init=weight_init, activation=activation, bn=bn)
        self.res3 = ResBasicBlock(out_channel // 2, out_channel, 3,
                                  weight_init=weight_init, activation=activation, bn=bn)
    
    
    def forward(self, x, noise):
        out = self.res1(torch.cat([x, noise], 1))  # concat noise
        out = self.res2(out)
        out = self.res3(out)
        return out
    
    
class DecodeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, weight_init=True, activation='relu', bn=True):
        super(DecodeBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = CBR(in_channel, out_channel, 3, stride=1, padding=1,
                         weight_init=weight_init, activation=activation, bn=bn)
        self.bridge = CBR(in_channel, out_channel, 3, stride=1, padding=1,
                        weight_init=weight_init, activation=activation, bn=bn)

        self.res = ResBasicBlock(out_channel * 2, out_channel, 3, stride=1, padding=1,  # out - out
                                 weight_init=weight_init, activation=activation, bn=bn)
        self.conv2 = CBR(out_channel, out_channel, 3, stride=1, padding=1,
                         weight_init=weight_init, activation=activation, bn=bn)

    def forward(self, front, rear):
        front = self.upsample(front)
        front = self.conv(front)
        rear = self.bridge(rear)
        merge = torch.cat([front, rear], dim=1)
        merge = self.res(merge)
        out = self.conv2(merge)
        return out



class Generator(nn.Module):
    # U-net
    def __init__(self):
        super(Generator, self).__init__()
        # size 256
        self.conv = CBR(in_channel=1, out_channel=16, kernel_size=5, stride=1, padding=2,
                        weight_init=True, activation='leaky')
        # encode stride=2
        # size 256
        self.conv1 = EncodeBlock(16, 32, 5, stride=2, padding=2, activation='leaky')
        # size 128
        self.conv2 = EncodeBlock(32, 64, 5, stride=2, padding=2, activation='leaky')
        # size 64
        self.conv3 = EncodeBlock(64, 128, 3, stride=2, padding=1, activation='leaky')
        # size 32
        self.conv4 = EncodeBlock(128, 256, 3, stride=2, padding=1, activation='leaky')
        # size 16
        self.conv5 = EncodeBlock(256, 256, 3, stride=2, padding=1, activation='leaky')
        # feature
        self.features = Feature(256, 256)

        # decode stride=1 scale_factor=2
        # size 8
        self.deconv5 = DecodeBlock(256, 128, activation='leaky')
        # size 16
        self.deconv4 = DecodeBlock(128, 64, activation='leaky')
        # size 32
        self.deconv3 = DecodeBlock(64, 32, activation='leaky')
        # size 64
        self.deconv2 = DecodeBlock(32, 16, activation='leaky')
        # size 128
        self.deconv1 = DecodeBlock(16, 8, activation='leaky')
        # size 256
        # to image
        self.outconv1 = CBR(8, 3, 3, stride=1, padding=1, weight_init=True, activation='leaky')
        self.outconv2 = CBR(3, 3, 3, stride=1, padding=1, weight_init=False, bn=False, activation='')

    def forward(self, x, noise):
        # encode
        conv = self.conv(x)  # 16
        conv1 = self.conv1(conv)  # 32
        conv2 = self.conv2(conv1)  # 64
        conv3 = self.conv3(conv2)  # 128
        conv4 = self.conv4(conv3)  # 256
        conv5 = self.conv5(conv4)  # 256

        feature = self.features(conv5, noise)

        deconv5 = self.deconv5(feature, conv4)
        deconv4 = self.deconv4(deconv5, conv3)
        deconv3 = self.deconv3(deconv4, conv2)
        deconv2 = self.deconv2(deconv3, conv1)
        deconv1 = self.deconv1(deconv2, conv)

        out = self.outconv1(deconv1)
        out = self.outconv2(out)

        return out




class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv0 = CBR(3, 32, 3, stride=2, padding=1, activation='leaky')
        self.conv1 = CBR(32, 64, 3, stride=2, padding=1, activation='leaky')
        self.conv2 = CBR(64, 128, 3, stride=2, padding=1, activation='leaky')

        self.conv3 = ResBasicBlock(128, 256, 3, stride=1, padding=1, activation='leaky')
        self.conv4 = ResBasicBlock(256, 256, 3, stride=1, padding=1, activation='leaky')

        self.conv5 = CBR(256, 1, 3, stride=1, padding=1, weight_init=False, bn=False, activation='')


    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)

        out = self.conv3(out)
        out = self.conv4(out)

        out = self.conv5(out)

        return out

