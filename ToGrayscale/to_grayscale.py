import cv2
import numpy as np
import glob, os

dirpath_dataset = '../dataset/'

imgpaths = glob.glob(dirpath_dataset + '**/*.png', recursive=True)

print(str(len(imgpaths)) + ' images listed.')

ksize = 5

for imgpath in imgpaths:
    imgpath = imgpath.replace('\\', '/')

    dirpath = '/'.join(imgpath.split('/')[:-1])
    imgname = imgpath.split('/')[-1][:-len('.png')]
    imgsize = int(imgpath.split('/')[-4][5:])

    if imgname[-8:-2] == '_sigma':
        continue
 
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(imgpath, img)

    h, w = img.shape

    img_expand = np.zeros((h + ksize - 1, w + ksize - 1), img.dtype)
    img_expand[ksize//2:-ksize//2+1,ksize//2:-ksize//2+1] = img
    img_expand = img_expand.astype(np.float32)

    for suffix, sigma in zip(['_sigma05', '_sigma10', '_sigma15', '_sigma20'], [0.5, 1, 1.5, 2.0]):
        img_blur = cv2.GaussianBlur(img_expand, (ksize, ksize), sigma)
        img_blur = (img_blur - np.min(img_blur)) / (np.max(img_blur) - np.min(img_blur)) * 255
        img_blur = img_blur[ksize//2:-ksize//2+1,ksize//2:-ksize//2+1]

        cv2.imwrite(dirpath + '/' + imgname + suffix + '.png', img_blur)

    print('finished ' + imgpath)
