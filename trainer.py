import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from dice import multiclass_dice_coeff, dice_coeff
from metrics import *

class Trainer:
    def __init__(self, model, trainloader, validloader, epochs, criterion, optimizer, scheduler, device, savepath, savename):
        
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
            # mask = F.one_hot(mask, self.model.n_classes)    # (batch, h, w, classes)
            # # mask = mask.permute(0, 3, 1, 2).float()     
            # print(mask.shape)
            # print(pred_mask.shape)
            # compute the loss
            loss = self.criterion(pred_mask, mask)
            # loss = cross_entropy2d(pred_mask, mask.long(), reduction='mean')
            # back propagation
            loss.backward()
            # parameter update
            self.optimizer.step()

            tr_losses.append(loss.cpu().detach().numpy())

            
            
            # assert mask.min() >= 0 and mask.max() < self.model.n_classes, 'True mask indices should be in [0, n_classes]'
            # convert to one-hot
            # mask = F.one_hot(mask, self.model.n_classes)    # (batch, h, w, classes)
            # mask = mask.permute(0, 3, 1, 2).float()         # (batch, classes, h, w)
            # pred_mask = F.one_hot(pred_mask.argmax(dim=1), self.model.n_classes)
            # print(pred_mask.shape)
            # pred_mask = mask.permute(0, 3, 1, 2).float()
            # print(pred_mask.shape)
            # compute dice score
            # dice_score = multiclass_dice_coeff(pred_mask[:, 1:], mask[:, 1:])
            # tr_dice_scores.append(dice_score)

            avg_tr_loss = sum(tr_losses) / len(tr_losses)
            # avg_dice_score = sum(tr_dice_scores) / len(tr_dice_scores)
            avg_dice_score = 0
        return avg_tr_loss, avg_dice_score
        
    def valid_fn(self):
        self.model.eval()
        val_losses = []
        val_dice_scores = []

        with torch.no_grad():
            for img, mask in tqdm(self.valid_loader):
                img = img.to(self.device)
                mask = mask.to(self.device)
                # img = img.cuda()
                # mask = mask.cuda()

                # size = mask.size()
                # print(size)
                # mask = mask[:, 0, :, :].view(size[0], 1, size[2], size[3])
                # print(mask.shape)
                # oneHot_size = (size[0], 19, size[2], size[3])
                # mask = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
                # mask = mask.scatter_(1, mask.data.long().cuda(), 1.0)
                # print(mask.shape)
                
                pred_mask = self.model(img)
                loss = self.criterion(pred_mask, mask)
                # loss = cross_entropy2d(pred_mask, mask.long(), reduction='mean')
                val_losses.append(loss.cpu().detach().numpy())

                # assert mask.min() >= 0 and mask.max() < self.model.n_classes, 'True mask indices should be in [0, n_classes]'
                # convert to one-hot
                # mask = F.one_hot(mask, self.model.n_classes)    # (batch, h, w, classes)
                # mask = mask.permute(0, 3, 1, 2).float()         # (batch, classes, h, w)
                # pred_mask = F.one_hot(pred_mask.argmax(dim=1), self.model.n_classes)
                # pred_mask = mask.permute(0, 3, 1, 2).float()
                # compute dice score
                # dice_score = multiclass_dice_coeff(pred_mask[:, 1:], mask[:, 1:])
                # val_dice_scores.append(dice_score)


            avg_val_loss = sum(val_losses) / len(val_losses)
            # avg_dice_score = sum(val_dice_scores) / len(val_dice_scores)
            avg_dice_score = 0
            return avg_val_loss, avg_dice_score
        
    def run(self):
        history = {
        'train_loss' : [],
        'train_dice' : [],
        'valid_loss' : [],
        'valid_dice' : []
        }

        for epoch in range(self.epochs):
            print(f'Epoch {epoch+1:03d} / {self.epochs:03d}:')
            train_loss, train_dice = self.train_fn()
            valid_loss, valid_dice = self.valid_fn()
            self.scheduler.step(valid_dice)

            print('lr:',self.get_lr(self.optimizer))
            print(f'train loss: {train_loss}  valid loss: {valid_loss}')
            print(f'train dice: {train_dice}  valid dice: {valid_dice}')

            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
            history['train_dice'].append(train_dice)
            history['valid_dice'].append(valid_dice)

            # save model if best valid
            if torch.tensor(history['valid_loss']).argmin() == epoch:
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
        ax[1].plot(range(self.epochs), metrics['valid_dice'], label='Valid')
        ax[1].legend()
        plt.show()
        fig.savefig(str(self.savepath / 'metrics.jpg'))
        plt.close()

    
