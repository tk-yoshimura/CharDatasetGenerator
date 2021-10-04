import cv2
import glob, os

dirpath = '../dataset/'

imgpaths = glob.glob(dirpath + '**/*.png', recursive=True)

print(str(len(imgpaths)) + ' images listed.')

for imgpath in imgpaths:
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(imgpath, img)

    print('finished ' + imgpath)
