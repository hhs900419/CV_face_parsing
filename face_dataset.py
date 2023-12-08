import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json
import cv2

class CelebAMask_HQ_Dataset(Dataset):
    def __init__(self, 
                 root_dir,
                 sample_indices, 
                 mode,
                 tr_transform=None):

        assert mode in ('train', "val", "test")

        self.root_dir = root_dir
        self.mode = mode

        self.tr_transform = tr_transform
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.image_dir = os.path.join(root_dir, 'CelebA-HQ-img')  # Path to image folder
        self.mask_dir = os.path.join(root_dir, 'mask')    # Path to mask folder
        self.sample_indices = sample_indices

        # self.images = os.listdir(self.image_dir)
        # self.masks = os.listdir(self.mask_dir)
        
        self.train_dataset = []
        self.test_dataset = []
        self.preprocess()
        
        # if train_mode:
        #     self.num_images = len(self.train_dataset)
        # else:
        #     self.num_images = len(self.test_dataset)
        
        

    def preprocess(self):
        for i in range(len([name for name in os.listdir(self.image_dir) if osp.isfile(osp.join(self.image_dir, name))])):
            img_path = osp.join(self.image_dir, str(i)+'.jpg')
            label_path = osp.join(self.mask_dir, str(i)+'.png')

            if self.mode != "test":
                self.train_dataset.append([img_path, label_path])
            else:
                self.test_dataset.append([img_path, label_path])

    def __getitem__(self, idx):
        idx = self.sample_indices[idx]
        
        if self.mode != "test":
            img_pth, mask_pth = self.train_dataset[idx]
        else:
            img_pth, mask_pth = self.test_dataset[idx]
        
        

        # read img, mask
        image = Image.open(img_pth).convert('RGB')
        image = image.resize((512, 512), Image.BILINEAR)
        # mask = Image.open(mask_pth).convert('P')
        mask = Image.open(mask_pth).convert('L')
        
        # mask = Image.open(mask_pth).convert('L')

        # data augmentation
        if self.mode == 'train':
            image, mask = self.tr_transform(image, mask)
            # mask = self.tr_transform(mask)

        image = self.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        # mask = np.array(mask).astype(np.int64)[np.newaxis, :]
        # mask = transforms.ToTensor()(mask)
        
        
        return image, mask


    def __len__(self):
        return len(self.sample_indices)