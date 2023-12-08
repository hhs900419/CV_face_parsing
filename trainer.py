import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from dice import multiclass_dice_coeff, dice_coeff
from metrics import *
from criterion import *

class Trainer:
    def __init__(self, model, trainloader, validloader, epochs, criterion, optimizer, device, savepath, savename, scheduler=None):
        
        self.model = model
        self.train_loader = trainloader
        self.valid_loader = validloader

        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.savepath = savepath
        self.savename = savename

        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

    def train_fn(self):
        self.model.train()
        tr_losses = []
        tr_dice_scores = []

        for img, mask in tqdm(self.train_loader):
            img = img.to(self.device)
            mask = mask.to(self.device)
            # img = img.cuda()
            # mask = mask.cuda()
            
            h, w = mask.size()[1], mask.size()[2]

            # size = mask.size()
            # print(size)
            # mask = mask[:, 0, :, :].view(size[0], 1, size[2], size[3])
            # print(mask.shape)
            # oneHot_size = (size[0], 19, size[2], size[3])
            # mask = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            # mask = mask.scatter_(1, mask.data.long().cuda(), 1.0)
            # print(mask.shape)
            # print(mask.min())
            # print(mask.max())

            # 1. clear gradient
            self.optimizer.zero_grad()
            # 2. forward
            pred_mask = self.model(img)
            # compute the loss
            loss1 = self.criterion(pred_mask, mask)
            loss2 = cross_entropy2d(pred_mask, mask.long(), reduction='mean')
            loss = loss1 + loss2
            # back propagation
            loss.backward()
            # parameter update
            self.optimizer.step()

            tr_losses.append(loss.cpu().detach().numpy())

            avg_tr_loss = sum(tr_losses) / len(tr_losses)
            # avg_dice_score = sum(tr_dice_scores) / len(tr_dice_scores)
            avg_dice_score = 0
        return avg_tr_loss, avg_dice_score
        
    def valid_fn(self):
        self.model.eval()
        val_losses = []
        val_dice_scores = []
        
        metrics = SegMetric(n_classes=19)

        with torch.no_grad():
            for img, mask in tqdm(self.valid_loader):
                img = img.to(self.device)
                mask = mask.to(self.device)
                # img = img.cuda()
                # mask = mask.cuda()
                
                h, w = mask.size()[1], mask.size()[2]

                # size = mask.size()
                # print(size)
                # mask = mask[:, 0, :, :].view(size[0], 1, size[2], size[3])
                # print(mask.shape)
                # oneHot_size = (size[0], 19, size[2], size[3])
                # mask = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
                # mask = mask.scatter_(1, mask.data.long().cuda(), 1.0)
                # print(mask.shape)
                
                outputs = self.model(img)
                loss1 = self.criterion(outputs, mask)
                loss2 = cross_entropy2d(outputs, mask.long(), reduction='mean')
                loss = loss1 + loss2
                val_losses.append(loss.cpu().detach().numpy())
                
                outputs = F.interpolate(outputs, (h, w), mode='bilinear', align_corners=True)
                pred = outputs.data.max(1)[1].cpu().numpy()  # Matrix index
                gt = mask.cpu().numpy()
                metrics.update(gt, pred)

            avg_val_loss = sum(val_losses) / len(val_losses)
            # avg_dice_score = sum(val_dice_scores) / len(val_dice_scores)
            metric_scores = metrics.get_scores()[0]
            return avg_val_loss, metric_scores
        
    def run(self):
        history = {
        'train_loss' : [],
        'train_dice' : [],
        'valid_loss' : [],
        'valid_miou' : []
        }

        for epoch in range(self.epochs):
            print(f'Epoch {epoch+1:03d} / {self.epochs:03d}:')
            train_loss, train_dice = self.train_fn()
            valid_loss, metric_score = self.valid_fn()
            val_miou = metric_score["Mean IoU : \t"]
            if self.scheduler:
                self.scheduler.step(val_miou)
            
            print('lr:',self.get_lr(self.optimizer))
            print(f'train loss: {train_loss}  valid loss: {valid_loss}')
            # print(f'train dice: {train_dice}  valid dice: {valid_dice}')
            print("----------------- Total Val Performance --------------------")
            for k, v in metric_score.items():
                print(k, v)
            print("---------------------------------------------------")

            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
            history['train_dice'].append(train_dice)
            history['valid_miou'].append(val_miou)

            # save model if best valid
            # if torch.tensor(history['valid_loss']).argmin() == epoch:
            if torch.tensor(history['valid_miou']).argmax() == epoch:
                torch.save(self.model.state_dict(), os.path.join(self.savepath, self.savename))
                print('Model Saved!')

        self.plot_save_history(history)


    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    def plot_save_history(self, metrics):
        # Plot the loss curve against epoch
        fig, ax = plt.subplots(2, 1, figsize=(10, 10), dpi=100)
        ax[0].set_title('Loss (L1)')
        ax[0].plot(range(self.epochs), metrics['train_loss'], label='Train')
        ax[0].plot(range(self.epochs), metrics['valid_loss'], label='Valid')
        ax[0].legend()
        ax[1].set_title('MSE')
        ax[1].plot(range(self.epochs), metrics['train_dice'], label='Train')
        ax[1].plot(range(self.epochs), metrics['valid_miou'], label='Valid')
        ax[1].legend()
        plt.show()
        fig.savefig(os.path.join(self.savepath , 'metrics.jpg'))
        plt.close()

    
