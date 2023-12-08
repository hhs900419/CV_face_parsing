import os.path as osp
import os
import cv2
import numpy as np
# from transform import *
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from configs import *


def preprocess():
    configs = Configs()
    face_data = os.path.join(configs.root_dir, 'CelebA-HQ-img')
    face_sep_mask = os.path.join(configs.root_dir, 'CelebAMask-HQ-mask-anno')
    mask_path = os.path.join(configs.root_dir, 'mask')

    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    counter = 0
    total = 0

    # for i in range(1):
    for i in tqdm(range(15)):

        atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

        for j in tqdm(range(i * 2000, (i + 1) * 2000)):

            mask = np.zeros((512, 512))

            # index starts from 1
            for l, att in enumerate(atts, 1):
                total += 1
                file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
                path = osp.join(face_sep_mask, str(i), file_name)

                if os.path.exists(path):
                    counter += 1
                    sep_mask = np.array(Image.open(path).convert('P'))
                    # print(np.unique(sep_mask))
                    mask[sep_mask == 225] = l
            cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)


    print(counter, total)