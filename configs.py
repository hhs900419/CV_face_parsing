import torch

class Configs():
    def __init__(self):
        self.seed = 1187
        self.root_dir = "/home/hsu/HD/dataset/CelebAMask-HQ"
        self.val_size = 0.15
        self.batch_size = 6
        self.n_workers = 4
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epochs = 15
        self.lr = 5e-4
        self.model_path = './model_weight/'
        self.cmp_result_dir = './result'
        

        