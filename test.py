import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
from dataloader import list_images, load_dataset, get_train_images
#from net import pf_net
import loss_func
from torch.utils.tensorboard import SummaryWriter
from imageio import imread, imsave
import random
import shutil
from swin_fusion import swin_fusion_net
EPSILON = 1e-5
import numpy as np
from torchvision import transforms

def test(image_set_ir):

    #criterion = loss_func.pf_loss().cuda()
    #image_set_ir, batches = load_dataset(data_iter, 32)
    #pfnet.eval()
    #pfnet.cuda()
    count = 0

    #image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]

    img_ir = get_train_images(image_set_ir)

    image_paths_p = [x.replace('ir', 'polar_p') for x in image_set_ir]
    img_p = get_train_images(image_paths_p)

    img_ir = (img_ir - torch.min(img_ir)) / (torch.max(img_ir) - torch.min(img_ir) + EPSILON)
    img_p = (img_p - torch.min(img_p)) / (torch.max(img_p) - torch.min(img_p) + EPSILON)

    # img_ir = img_ir.cuda()
    # img_p = img_p.cuda()

    #img_ir = img_ir[:,:,64:,96:-96]
    #img_p = img_p[:,:,64:,96:-96]

    #output_pf = pfnet(img_ir, img_p)

    #loss = criterion(img_ir, img_p, output_pf)


    return img_ir,img_p


if __name__ == "__main__":

    test_path = 'dataset/test/ir/'
    pfnet = swin_fusion_net()
    #pfnet = pf_net()
    pfnet.load_state_dict(torch.load('model/model.pth'))  #save_pth/2022_2_28/Epoch3_loss0.180.pth
    #pfnet.cuda()
    pfnet.eval()
    test_imgs_path = list_images(test_path)
    img_ir, img_p = test(test_imgs_path)
    #s=img_ir[0].unsqueeze(1).shape
    for ii in range(img_ir.shape[0]):
       output_pf, _, _ = pfnet(img_ir[ii].unsqueeze(0), img_p[ii].unsqueeze(0))
       out = output_pf.detach().cpu().numpy().squeeze()
       out = (out - np.min(out)) / (np.max(out) - np.min(out))
       imsave(test_path.replace('ir', 'fusion')+test_imgs_path[ii].split('/')[-1],(out*255).astype('uint8'))
    #torchvision.utils.save_image(out_pf, test_path.replace('ir', 'result') + test_imgs_path[0].split('/')[-1])
    #print(out_pf.shape)













