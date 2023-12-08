import os.path as osp
import os
import cv2
import numpy as np
from augmentation import *
from face_dataset import *
from unet import *
from criterion import *
from trainer import *
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.backends import cudnn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import optim
import gc


def train():
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    torch.cuda.manual_seed(2020)
    
    
    ### Train/Val/Test Split ###
    root_dir = "/home/hsu/HD/dataset/CelebAMask-HQ"
    image_dir = os.path.join(root_dir, 'CelebA-HQ-img')

    train_indices = set()
    indices_file_pth = os.path.join(root_dir, 'train.txt')
    with open(indices_file_pth, 'r') as file:
        train_indices = set(map(int, file.read().splitlines()))
        
    sample_indices = list(range(len(os.listdir(image_dir))))
    test_indices = [idx for idx in sample_indices if idx not in train_indices]
    # Split indices into training and validation sets
    train_indices = list(train_indices)
    # train_indices = train_indices[:100]         ############################################################################################################
    train_indices, valid_indices = train_test_split(train_indices, test_size=0.15, random_state=1187)
    print(len(train_indices))
    print(len(valid_indices))
    print(len(test_indices))
    # print(test_indices)
    
    # augmentation
    train_tranform = Compose({
        RandomCrop(448),
        RandomHorizontallyFlip(p=0.5),
        AdjustBrightness(bf=0.1),
        AdjustContrast(cf=0.1),
        AdjustHue(hue=0.1),
        AdjustSaturation(saturation=0.1)
    })
    
    ### dataset ###
    trainset = CelebAMask_HQ_Dataset(root_dir=root_dir, 
                                sample_indices=train_indices,
                                mode='train', 
                                tr_transform=train_tranform)
    validset = CelebAMask_HQ_Dataset(root_dir=root_dir, 
                                    sample_indices=valid_indices, 
                                    mode = 'val')
    
    
    ### dataloader ###
    batch_size = 6
    n_workers = 4

    # sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    train_loader = DataLoader(trainset,
                        batch_size = batch_size,
                        shuffle = True,
                        # sampler = sampler,
                        num_workers = n_workers,
                        pin_memory = True,
                        drop_last = True)

    valid_loader = DataLoader(validset,
                        batch_size = batch_size,
                        shuffle = False,
                        num_workers = n_workers, 
                        pin_memory = True,
                        drop_last = True)
    print(f"training data{len(train_indices)} and validation data{len(valid_indices)} loaded succesfully ...")
    
    gc.collect()
    torch.cuda.empty_cache()    
    
    ### Init model ###
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Unet(n_channels=3, n_classes=19).to(DEVICE)
    print("Model Initialized !")
    
    ### hyper params ###
    EPOCHS = 15
    LR = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    criterion = DiceLoss()
    # criterion = dice_loss
    SAVEPATH = './model_weight/'
    SAVENAME = f'model_aug_{EPOCHS}eps.pth'
    
    ### training ###
    Trainer( model=model, 
        trainloader=train_loader,
        validloader=valid_loader,
        epochs=EPOCHS,
        criterion=criterion, 
        optimizer=optimizer,
        scheduler=scheduler, 
        device=DEVICE,
        savepath=SAVEPATH, 
        savename=SAVENAME).run()
    
    
if __name__ == "__main__":
    train()
    

    
    