import cv2
import glob, os
import numpy as np

dirpath = '../dataset/'

def make_npz(imgsize, category, chars):
    imgpaths = glob.glob(dirpath + '{}/size_{}/**/*.png'.format(category, imgsize), recursive=True)

    dataset = {}
    for c in chars:
        dataset[c] = []

    for imgpath in imgpaths:
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    
        for i, c in enumerate(chars):
            img_split = img[:, i * imgsize : (i + 1) * imgsize]
            dataset[c].append(img_split)

        print('appended: ' + imgpath)

    for c in chars:
        dataset[c] = np.stack(dataset[c], axis = 0)
    
    np.savez_compressed(dirpath + category + '_size_{}.npz'.format(imgsize), **dataset)

for imgsize in [16]:
    make_npz(imgsize, category='numeric',  chars='0123456789')
    make_npz(imgsize, category='alphabet', chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-')