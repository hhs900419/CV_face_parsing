import torch

class Configs():
    def __init__(self):
        self.seed = 1187
        # self.root_dir = "/home/hsu/HD/dataset/CelebAMask-HQ"
        self.root_dir = "/home/hsu/HD/CV/CelebAMask-HQ"
        self.val_size = 0.15
        self.batch_size = 8
        self.n_workers = 6
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epochs = 20
        self.lr = 5e-4
        # self.model_path = './model_weight/'
        self.model_path = '../model_weight/'
        self.model_weight = f'model_aug_miou_{self.epochs}eps.pth'
        # self.model_weight = f'model_debug.pth'
        self.cmp_result_dir = './result'
        self.debug = False
        

        