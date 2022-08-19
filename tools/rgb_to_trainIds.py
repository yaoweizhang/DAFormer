import os
import cv2
import numpy as np
import glob
import argparse

CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
            'bicycle')

PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
           [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
           [0, 80, 100], [0, 0, 230], [119, 11, 32]]

label_dict = np.ones(256*256*256, dtype='uint8') * 255

for i, rgb in enumerate(PALETTE):
    label_dict[rgb[0]*256*256 + rgb[1]*256 + rgb[2]] = i

def convt(work_dir):
    imgs = glob.glob(work_dir + '/preds/**/*.png', recursive=True)
    new_fold = os.path.join(work_dir, 'labelTrainIds')
    os.mkdir(new_fold)
    for img_path in imgs:
        img = cv2.imread(img_path, -1)
        img_name = img_path.split('/')[-1]
        #img is in bgr mode
        img = img.astype('uint32')
        temp = img[:, :, 0] + img[:, :, 1]*256 + img[:, :, 2] * 256 * 256
        new_img = label_dict[temp]
        new_path = os.path.join(new_fold, img_name[:-12] + 'gt_labelTrainIds.png')
        cv2.imwrite(new_path, new_img)

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default=None, help='work dir for job')
args = parser.parse_args()

convt(args.dir)
