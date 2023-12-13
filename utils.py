import os
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn


def make_folder(path, version):
    if not osp.exists(osp.join(path, version)):
        os.makedirs(osp.join(path, version))


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])


def labelcolormap(N):
    if N == 19:  # CelebAMask-HQ
        cmap = np.array([(0,  0,  0), (204, 0,  0), (76, 153, 0),
                         (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
                         (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
                         (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153),
                         (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


class Colorize(object):
    def __init__(self, n=19):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def tensor2label(label_tensor, n_label, imtype=np.uint8):
    label_tensor = label_tensor.cpu().float()
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = label_tensor.numpy()
    label_numpy = label_numpy / 255.0
    return label_numpy


def generate_label(inputs, imsize, class_num=19):
    '''Tensor after optimized...'''

    inputs = F.interpolate(input=inputs, size=(imsize, imsize),
                           mode='bilinear', align_corners=True)
    pred_batch = torch.argmax(inputs, dim=1)
    label_batch = torch.Tensor(
        [tensor2label(p.view(1, imsize, imsize), class_num) for p in pred_batch])
    return label_batch

def generate_label_plain(inputs, imsize, class_num=19):
    inputs = F.interpolate(input=inputs, size=(imsize, imsize),
                           mode='bilinear', align_corners=True)
    pred_batch = torch.argmax(inputs, dim=1)
    label_batch = [p.cpu().numpy() for p in pred_batch]
    return label_batch

def generate_compare_results(images, labels, preds, imsize, class_num=19):
    '''Tensor after optimized...'''
    labels = F.interpolate(input=labels, size=(imsize, imsize),
                           mode='bilinear', align_corners=True)
    label_batch = torch.argmax(labels, dim=1)
    labels_batch = torch.Tensor(
        [tensor2label(p.view(1, imsize, imsize), class_num) for p in label_batch])
    preds = F.interpolate(input=preds, size=(imsize, imsize),
                           mode='bilinear', align_corners=True)
    pred_batch = torch.argmax(preds, dim=1)
    preds_batch = torch.Tensor(
        [tensor2label(p.view(1, imsize, imsize), class_num) for p in pred_batch])
    compare_batch = torch.cat((denorm(images).cpu().data, labels_batch, preds_batch), 3)
    return compare_batch


def adjust_learning_rate(g_lr, optimizer, i_iter, total_iters):
    """The learning rate decays exponentially"""

    def lr_poly(base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    lr = lr_poly(g_lr, i_iter, total_iters, .9)
    optimizer.param_groups[0]['lr'] = lr

    return lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
    

    # This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

labels_celeb = ['background','skin','nose',
        'eye_g','l_eye','r_eye','l_brow',
        'r_brow','l_ear','r_ear','mouth',
        'u_lip','l_lip','hair','hat',
        'ear_r','neck_l','neck','cloth']
def read_mask(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if type(img) is type(None):
        return np.zeros((256, 256, 1), dtype=np.uint8)
    return img

def mask2binary(path):
    mask = read_mask(path)
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask = np.where(mask > 0,1,0)
    return mask

def rle_encode(img): 
    pixels = img.flatten()
    if np.sum(pixels)==0:
        return '0'
    pixels = np.concatenate([[0], pixels, [0]]) 
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1 
    runs[1::2] -= runs[::2]
    # to string sep='_'
    runs = '_'.join(str(x) for x in runs)
    return runs

def rle_decode(mask_rle, shape): 
    s = mask_rle.split('_')
    s = [0 if x=='' else int(x) for x in s]
    if np.sum(s)==0:
        return np.zeros(shape, dtype=np.uint8)
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])] 
    starts -= 1 
    ends = starts + lengths 
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8) 
    for lo, hi in zip(starts, ends): 
         img[lo:hi] = 255
    return img.reshape(shape)

def mask2csv(mask_paths, csv_path='mask.csv',image_id=1,header=False):
    """
        mask_paths: dict of label:mask_paths
        ['label1':path1,'label2':path2,...]
    """
    results = []
    for i, label in enumerate(labels_celeb):
        try:
            mask = mask2binary(mask_paths[label])
        except:
            print("132456748974641")
            mask = np.zeros((256, 256), dtype=np.uint8)
        mask = rle_encode(mask)
        results.append(mask)
    df = pd.DataFrame(results)
    df.insert(0,'label',labels_celeb)
    df.insert(0,'Usage',["Public" for i in range(len(results))])
    df.insert(0,'ID',[image_id*19+i for i in range(19)])
    if header:
        df.columns = ['ID','Usage','label','segmentation']
    # print()
    # print(df)
    df.to_csv(csv_path,mode='a',header=header,index=False)

def mask2csv2(masks, csv_path='mask.csv',image_id=1,header=False):
    """
        mask_paths: dict of label:mask
        ['label1':mask1,'label2':mask2,...]
    """
    results = []
    for i, label in enumerate(labels_celeb):
        try:
            mask = masks[label]
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        except:
            mask = np.zeros((256, 256), dtype=np.uint8)
        mask = rle_encode(mask)
        results.append(mask)
    df = pd.DataFrame(results)
    df.insert(0,'label',labels_celeb)
    df.insert(0,'Usage',["Public" for i in range(len(results))])
    df.insert(0,'ID',[image_id*19+i for i in range(19)])
    
    if header:
        df.columns = ['ID','Usage','label','segmentation']
    # print()
    # print(df)
    df.to_csv(csv_path,mode='a',header=header,index=False)
    
def one_hot_encode(segmentation_map, num_classes=19):
    # Create an empty array with dimensions (num_classes, height, width)
    one_hot_mask = np.zeros((num_classes, segmentation_map.shape[0], segmentation_map.shape[1]), dtype=np.uint8)
    
    # Iterate through each class and set the corresponding channel to 1
    for class_label in range(num_classes):
        one_hot_mask[class_label, :, :] = (segmentation_map == class_label).astype(np.uint8)
    
    return one_hot_mask