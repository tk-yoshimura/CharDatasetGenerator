import random
import numpy as np

shiftsize = 2
shiftexpands = shiftsize * 2

def shift(in_data):
    img, label = in_data

    c, h, w = img.shape
    y, x = random.randrange(shiftexpands + 1), random.randrange(shiftexpands + 1)

    img_expand = np.zeros((c, h + shiftexpands, w + shiftexpands), img.dtype)
    img_expand[:, y:y+h, x:x+w] = img

    img_crop = img_expand[:, shiftsize:shiftsize+h, shiftsize:shiftsize+w]

    return img_crop, label

