import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2
from os import listdir
from imageio import imread
from os.path import join
random.seed(1143)


def load_data(img):
    #rr = ['I000', 'I045', 'I090', 'I135']
    files = os.listdir(img)
    img_list = []
    for k in files:
        img_list_tmp = [os.path.join(img, k, k + '_I.png'), os.path.join(img, k, k + '_dop.png')]
        #img_tmp = np.stack([im_I, im_dop])
        img_list.append(img_list_tmp)
    #data_list = np.stack(img_list).astype('float32')
    #data_list = data_list / 255.0

    # train_list = torch.from_numpy(train_list)
    # train_list = train_list.permute(0,1,4,2,3)
    # img_data = train_list[:,0:4,:,:,:]
    # img_labels = train_list[:,4:8,:,:,:]

    return img_list #torch.from_numpy(data_list)


class Dataset_load(data.Dataset):  # 继承Dataset
    def __init__(self, data):
        self.data = data  # 训练集
        #self.label = label  # 训练输出

    def __len__(self):  # 返回整个数据集的大小  图像的垂直尺寸（高度）   图像的水平尺寸（宽度）
        return len(self.data)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        h = random.randint(128, 400)
        w = random.randint(128, 564)

        I_index = torch.from_numpy(imread(self.data[index][0])[h-32:h+32, w-32:w+32]).unsqueeze(0).type(torch.float32)
        dop_index = torch.from_numpy(imread(self.data[index][1])[h-32:h+32, w-32:w+32]).unsqueeze(0).type(torch.float32)

        # dop_index = self.data[index, 1, :, :].unsqueeze(0)
        # I_index = self.data[index, 0, :, :].unsqueeze(0)


        # label_index = self.label[index, :, :, h - 128:h + 128, w - 128:w + 128]
        # I_index = label_index[0, :, :, :] + label_index[2, :, :, :]
        # Q_index = label_index[0, :, :, :] - label_index[2, :, :, :]
        # U_index = label_index[1, :, :, :] - label_index[3, :, :, :]
        # dop_index = torch.div(torch.sqrt(torch.pow(Q_index, 2) + torch.pow(U_index, 2)), I_index + 0.0001)
        # aop_index = 0.5 * torch.atan(torch.div(U_index, Q_index))

        #sample = {'data': data_index, 'I': I_index, "Q": Q_index, "U": U_index}  # 根据图片和标签创建字典
        sample = { 'dop': dop_index/255.0, 'I': I_index/255.0 }

        return sample  # 返回该样本


def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    # print('BATCH SIZE %d.' % BATCH_SIZE)
    # print('Train images number %d.' % num_imgs)
    # print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        #print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches


def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file
        if name.endswith('.png'):
            images.append(join(directory, file))

        # name1 = name.split('.')
        names.append(name)
    return images #, names


def get_image(path, height=None, width=None, flag=False):
    if flag is True:
        image = Image.open(path).convert('RGB') #imread(path, mode='RGB')
    else:
        image = Image.open(path).convert('L')

    if height is not None and width is not None:
        #image = imresize(image, [height, width], interp='nearest')
        image = np.array(image.resize((height, width)))
        #image = image[image.shape[0]//4:, :]
    else:
        image = np.array(image)
        #image = image[image.shape[0]//4:, :]


    return image


def get_train_images(paths, height=None, width=None, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, flag)
        if flag is True:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.expand_dims(image, axis=0)  #np.reshape(image, [1, height, width])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images





if __name__ == "__main__":
    data_list = load_data('data')
    print(len(data_list))
