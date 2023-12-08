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
        cnt = 0

        with torch.no_grad():
            for img, mask in tqdm(self.test_loader):
                img = img.to(self.device)
                # labels = mask.to(self.device)
                mask = mask.to(self.device)
                
                labels = mask
                
                size = labels.size()
                h, w = size[1], size[2]

                pred_mask = self.model(img)
                loss = self.criterion(pred_mask, mask)
                # loss = cross_entropy2d(pred_mask, mask.long(), reduction='mean')
                test_losses.append(loss.cpu().detach().numpy())

                # print(f"pred: {pred_mask.shape}")
                # pred_mask = F.interpolate(pred_mask, (h, w), mode='bilinear', align_corners=True)
                # print(f"pred_interp: {pred_mask.shape}")
                # pred_mask = pred_mask.data.max(1)[1].cpu().numpy()  # Matrix index
                # print(f"pred_fin: {pred_mask.shape}")
                
                # print(f'lb: {labels.shape}')
                # labels = labels[:, :, :].view(size[0], 1, size[1], size[2])
                # print(f'lb_viewed: {labels.shape}')
                # oneHot_size = (size[0], 19, size[1], size[2])
                # labels_real = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
                # print(f'lb_oh: {labels_real.shape}')
                # labels_real = labels_real.scatter_(1, labels.data.long().cuda(), 1.0)
                # print(f'lb_oh_sc: {labels_real.shape}')

                # print(np.unique(pred_mask.data.max(1)[0].cpu().numpy()))
                pred_mask = pred_mask.data.max(1)[1].cpu().numpy()  # Matrix index
                mask = mask.cpu().detach().numpy()
                img = img.permute(0,2,3,1).cpu().detach().numpy()
                
                # cmap = np.array([(0,  0,  0), (204, 0,  0), (76, 153, 0),
                #          (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
                #          (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
                #          (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153),
                #          (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)],
                #         dtype=np.uint8)
                    
                # for im, pr_mask, gt_mask in zip(img, pred_mask, mask):
                #     print(pr_mask.shape)
                #     h, w = pr_mask.shape
                #     color_pr_mask = np.zeros((h, w, 3))
                #     color_gt_mask = np.zeros((h, w, 3))
                #     for label in range(0, len(cmap)):
                #         p_mask = (label == pr_mask)
                #         color_pr_mask[p_mask] = cmap[label]
                #         g_mask = (label == gt_mask)
                #         color_gt_mask[g_mask] = cmap[label]
                #     plt.figure(figsize=(12, 12))
                #     img_list = [im, color_pr_mask, color_gt_mask]

                #     if cnt < 50:
                #         for i in range(3):
                #             plt.subplot(1, 3, i+1)
                #             plt.imshow(img_list[i])
                #             plt.show()
                #         cnt += 1
                        
                
                        # color_image[0][mask] = self.cmap[label][0]
                        # color_image[1][mask] = self.cmap[label][1]
                        # color_image[2][mask] = self.cmap[label][2]
                    # print(im.shape)
                    # print(pr_mask.shape)
                    # print(gt_mask.shape)
                    
                    # print(np.unique(pr_mask))
                    # print(np.unique(gt_mask))
                    # print(np.sum(pr_mask == gt_mask))
                    # pass


            avg_test_loss = sum(test_losses) / len(test_losses)
            # avg_dice_score = sum(val_dice_scores) / len(val_dice_scores)
            avg_dice_score = 0
            return avg_test_loss, avg_dice_score
        
    def run(self):
        print(f'Evaluating ... ')
        test_loss, test_dice = self.test_fn()
        print(f'test loss: {test_loss}')




    