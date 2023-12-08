import os.path as osp
import os
import cv2
import numpy as np
from augmentation import *
from face_dataset import *
from unet import *
from criterion import *
from trainer import *
from configs import *
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.backends import cudnn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import optim
import gc


def train():
    configs = Configs()
    SEED = configs.seed
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    torch.cuda.manual_seed(SEED)
    
    
    ### Train/Val/Test Split ###
    ROOT_DIR = configs.root_dir
    image_dir = os.path.join(ROOT_DIR, 'CelebA-HQ-img')

    train_indices = set()
    indices_file_pth = os.path.join(ROOT_DIR, 'train.txt')
    with open(indices_file_pth, 'r') as file:
        train_indices = set(map(int, file.read().splitlines()))
        
    sample_indices = list(range(len(os.listdir(image_dir))))
    test_indices = [idx for idx in sample_indices if idx not in train_indices]
    # Split indices into training and validation sets
    train_indices = list(train_indices)
    # train_indices = train_indices[:100]         ############################################################################################################
    VAL_SIZE = configs.val_size
    train_indices, valid_indices = train_test_split(train_indices, test_size=VAL_SIZE, random_state=SEED)
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
    trainset = CelebAMask_HQ_Dataset(root_dir=ROOT_DIR, 
                                sample_indices=train_indices,
                                mode='train', 
                                tr_transform=train_tranform)
    validset = CelebAMask_HQ_Dataset(root_dir=ROOT_DIR, 
                                    sample_indices=valid_indices, 
                                    mode = 'val')
    
    
    ### dataloader ###
    BATCH_SIZE = configs.batch_size
    N_WORKERS = configs.n_workers

    # sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    train_loader = DataLoader(trainset,
                        batch_size = BATCH_SIZE,
                        shuffle = True,
                        # sampler = sampler,
                        num_workers = N_WORKERS,
                        pin_memory = True,
                        drop_last = True)

    valid_loader = DataLoader(validset,
                        batch_size = BATCH_SIZE,
                        shuffle = False,
                        num_workers = N_WORKERS, 
                        pin_memory = True,
                        drop_last = True)
    print(f"training data{len(train_indices)} and validation data{len(valid_indices)} loaded succesfully ...")
    
    gc.collect()
    torch.cuda.empty_cache()    
    
    ### Init model ###
    DEVICE = configs.device
    model = Unet(n_channels=3, n_classes=19).to(DEVICE)
    print("Model Initialized !")
    
    ### hyper params ###
    EPOCHS = configs.epochs
    LR = configs.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    criterion = DiceLoss()
    # criterion = dice_loss
    SAVEPATH = configs.savedir
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
    

    
    