import os.path as osp
import os
import cv2
import numpy as np
from augmentation import *
from face_dataset import *
from unet import *
from criterion import *
from tester import *
from configs import *
from utils import *
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.backends import cudnn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import optim
import gc


def test_fn():
    configs = Configs()
    SEED = configs.seed
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    torch.cuda.manual_seed(SEED)

    ### Train/Val/Test Split ###
    """
    create train/val/test index list (only test is used in this script)
    """
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
    if configs.debug:
        train_indices = train_indices[:100]         ############################################################################################################
    VAL_SIZE = configs.val_size
    train_indices, valid_indices = train_test_split(train_indices, test_size=VAL_SIZE, random_state=SEED)
    print(len(train_indices))
    print(len(valid_indices))
    print(len(test_indices))


    ### dataloader ###
    BATCH_SIZE = configs.batch_size
    N_WORKERS = configs.n_workers

    testset = CelebAMask_HQ_Dataset(root_dir=ROOT_DIR,
                                sample_indices=test_indices,
                                mode='test')

    test_loader = DataLoader(testset,
                        batch_size = BATCH_SIZE,
                        shuffle = False,
                        num_workers = N_WORKERS, 
                        pin_memory = True,
                        drop_last = True)
    
    ####################### for Debugging
    if configs.debug:
        validset = CelebAMask_HQ_Dataset(root_dir=ROOT_DIR, 
                                        sample_indices=valid_indices, 
                                        mode = 'val')
        valid_loader = DataLoader(validset,
                            batch_size = BATCH_SIZE,
                            shuffle = False,
                            num_workers = N_WORKERS, 
                            pin_memory = True,
                            drop_last = True)
    #################################
    

    ### load model ###
    DEVICE = configs.device
    SAVEPATH = configs.model_path
    OUTPUT_DIR = configs.cmp_result_dir
    MODEL_WEIGHT = configs.model_weight
    if configs.debug:
        MODEL_WEIGHT = 'model_debug.pth'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    criterion = DiceLoss()
    model = Unet(n_channels=3, n_classes=19).to(DEVICE)
    # model.load_state_dict(torch.load(os.path.join(SAVEPATH , 'model.pth')))
    model.load_state_dict(torch.load(os.path.join(SAVEPATH , MODEL_WEIGHT)))
    

    ### testing
    # Tester(model=model, 
    #    testloader=test_loader, 
    #    criterion=criterion, 
    #    device=DEVICE).run()
    
    labels_celeb = ['background','skin','nose',
        'eye_g','l_eye','r_eye','l_brow',
        'r_brow','l_ear','r_ear','mouth',
        'u_lip','l_lip','hair','hat',
        'ear_r','neck_l','neck','cloth']

    ### visualize
    cmap = np.array([(0,  0,  0), (204, 0,  0), (76, 153, 0),
                         (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
                         (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
                         (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153),
                         (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)],
                        dtype=np.uint8)
    
    to_tensor = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    image_dir = os.path.join(ROOT_DIR, 'CelebA-HQ-img') 
    mask_dir = os.path.join(ROOT_DIR, 'mask')    

    test_dataset =[]
    for i in range(len([name for name in os.listdir(image_dir) if osp.isfile(osp.join(image_dir, name))])):
        img_path = osp.join(image_dir, str(i)+'.jpg')
        label_path = osp.join(mask_dir, str(i)+'.png')
        test_dataset.append([img_path, label_path])

    # inference again in file order
    # for i in tqdm(range(0, len(train_indices))):
    for i in tqdm(range(0, len(test_indices))):
        idx = test_indices[i]
        if configs.debug:
            idx = valid_indices[i]
            idx = train_indices[i]
        img_pth, mask_pth = test_dataset[idx]

        image = Image.open(img_pth).convert('RGB')
        image = image.resize((512, 512), Image.BILINEAR)
        mask = Image.open(mask_pth).convert('L')

        image = to_tensor(image).unsqueeze(0)
        gt_mask = torch.from_numpy(np.array(mask)).long()

        pred_mask = model(image.to(DEVICE))     # predict
        pred_mask = pred_mask.data.max(1)[1].cpu().numpy()  # Matrix index  (1,19,h,w) => (1,h,w)
        
        image = image.squeeze(0).permute(1,2,0)     # (1,3,h,w) -> (h,w,3)
        pred_mask = pred_mask.squeeze(0)            # (1,h,w) -> (h,w)


        classes = ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
        one_hot_mask = one_hot_encode(pred_mask, 19)
        # print(one_hot_mask.shape)
        # print(one_hot_mask)
        
        test_dir = "test_result"
        TEST_ID_DIR = f'{test_dir}/Test-image-{idx}'
        if not os.path.exists(TEST_ID_DIR):
            os.makedirs(TEST_ID_DIR)

        dict_path = {}    

        for j in range(19):
            if j == 0:
                mask = one_hot_mask[j,:,:] * 0
            else:
                mask = one_hot_mask[j,:,:] * 255
            cv2.imwrite(f"{TEST_ID_DIR}/{classes[j]}.png", mask)
            dict_path[classes[j]] = f"{TEST_ID_DIR}/{classes[j]}.png"


        # generate color mask image
        color_gt_mask = cmap[gt_mask]
        color_pr_mask = cmap[pred_mask]
        
        # plt.figure(figsize=(13, 6))
        # image = Image.open(img_pth).convert('RGB')      # we want the image without normalization for plotting
        # image = image.resize((512, 512), Image.BILINEAR)
        # img_list = [image, color_pr_mask, color_gt_mask]
        # for i in range(3):
        #     plt.subplot(1, 3, i+1)
        #     plt.imshow(img_list[i])
        # plt.savefig(f"{OUTPUT_DIR}/result_{idx}.jpg")

        labels_celeb = ['background','skin','nose',
        'eye_g','l_eye','r_eye','l_brow',
        'r_brow','l_ear','r_ear','mouth',
        'u_lip','l_lip','hair','hat',
        'ear_r','neck_l','neck','cloth']

        right_order_mask_path = {}

        for lab in labels_celeb:
            right_order_mask_path[lab] = dict_path[lab]


        # print(i)
        mask2csv(mask_paths=right_order_mask_path, image_id=i)
        # break

if __name__ == "__main__":
    test_fn()


















