from math import exp
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#import TVLoss

from torchvision.transforms import ToPILImage


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map, sigma1_sq


def ssim(img1, img2, window_size=11, size_average=True):
    img1 = torch.clamp(img1, min=0, max=1)
    img2 = torch.clamp(img2, min=0, max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    value, sigma1_sq = _ssim(img1, img2, window, window_size, channel, size_average)
    v = torch.zeros_like(sigma1_sq) + 0.0001
    sigma1 = torch.where(sigma1_sq < 0.0001, v, sigma1_sq)
    return value, sigma1


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


# def SSIM_LOSS(img1, img2, size=11, sigma=1.5):
#     window = fspecial_gauss(size, sigma) # window shape [size, size]
#     K1 = 0.01
#     K2 = 0.03
#     L = 1
#     C1 = (K1*L)**2
#     C2 = (K2*L)**2
#     mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='SAME')
#     mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1], padding='SAME')
#     mu1_sq = mu1*mu1
#     mu2_sq = mu2*mu2
#     mu1_mu2 = mu1*mu2
#     sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='SAME') - mu1_sq
#     sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='SAME') - mu2_sq
#     sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='SAME') - mu1_mu2
#
#     v1 = 2*mu1_mu2+C1
#     v2 = mu1_sq+mu2_sq+C1
#
#     value = (v1*(2.0*sigma12 + C2))/(v2*(sigma1_sq + sigma2_sq + C2))
#
#     # sigma1_sq = sigma1_sq/(mu1_sq+0.00000001)
#     v = tf.zeros_like(sigma1_sq) + 0.0001
#     sigma1 = tf.where(sigma1_sq < 0.0001, v, sigma1_sq)
#     return value, sigma1

def func_loss(img1, img2, y):
    #img1, img2 = tf.split(y_, 2, 3)
    img3 = img1 * 0.5 + img2 * 0.5
    Win = [11, 9, 7, 5, 3]
    loss = 0
    for s in Win:
        loss1, sigma1 = ssim(img1, y, s)
        loss2, sigma2 = ssim(img2, y, s)
        r = sigma1 / (sigma1 + sigma2 + 0.0000001)
        tmp = 1 - torch.mean(r * loss1) - torch.mean((1 - r) * loss2)
        loss = loss + tmp
    loss = loss / 5.0
    loss = loss + torch.mean(torch.abs(img3 - y)) * 0.1
    return loss

class pf_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y1, y2, y, x1=0, x2=0):

        loss = func_loss(y1, y2, y)

        TV = TVLoss()

        loss = loss + 0.1*0.25*(TV(x1[0]-x2[0])+TV(x1[1]-x2[1])+TV(x1[2]-x2[2])+TV(x1[3]-x2[3]))

        #loss_aop = torch.mean(torch.abs(aop_t - aop))
        #loss_p = 0.5 * loss_dop + 0.5 * loss_aop
        return loss