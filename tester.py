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
from metrics import *

class Tester:
    def __init__(self, model,testloader, criterion, device):
        
        self.model = model
        self.test_loader = testloader
        self.criterion = criterion
        self.device = device

        
    def test_fn(self):
        self.model.eval()
        test_losses = []
        test_dice_scores = []
        metrics = SegMetric(n_classes=19)
        cnt = 0

        with torch.no_grad():
            for img, mask in tqdm(self.test_loader):
                img = img.to(self.device)
                mask = mask.to(self.device)
                
                labels = mask
                
                size = labels.size()
                h, w = size[1], size[2]

                outputs = self.model(img)
                loss1 = self.criterion(outputs, mask)
                loss2 = cross_entropy2d(outputs, mask.long(), reduction='mean')
                loss = loss1 + loss2
                test_losses.append(loss.cpu().detach().numpy())
                
                outputs = F.interpolate(outputs, (h, w), mode='bilinear', align_corners=True)
                pred = outputs.data.max(1)[1].cpu().numpy()  # Matrix index
                gt = mask.cpu().numpy()
                metrics.update(gt, pred)

            avg_test_loss = sum(test_losses) / len(test_losses)
            metric_scores = metrics.get_scores()[0]
            return avg_test_loss, metric_scores
        
    def run(self):
        print(f'Evaluating ... ')
        test_loss, metric_score = self.test_fn()
        print(f'test loss: {test_loss}')
        print("----------------- Total Test Performance --------------------")
        for k, v in metric_score.items():
            print(k, v)
        print("---------------------------------------------------")




    