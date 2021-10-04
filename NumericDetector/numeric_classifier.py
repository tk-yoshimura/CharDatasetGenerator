import cv2
import numpy as np

import chainer
from model import Model

imgsize = 16

model = Model()
model.load('../model/numeric_classifier_model.npz')

def normalize(img):
    """
    Parameters
    ---------
    img: np.ndarray
        2D array
        value = 0: background
        value > 0: character regions
    Returns
    ---------
    img_normalized: np.ndarray
        2D array (imgsize x imgsize)
        value = 0: background
        value = 1: character regions
    """

    img = img.astype(np.float32)
    h, w = img.shape
    
    scale = imgsize / max(w, h)
    offset_x, offset_y = (imgsize - w * scale) / 2, (imgsize - h * scale) / 2

    mat = np.array([[scale, 0, offset_x], [0, scale, offset_y]])

    img_warped = cv2.warpAffine(img, mat, (imgsize, imgsize), 
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    lmin, lmax = np.min(img_warped), np.max(img_warped)

    img_normed = (img_warped - lmin) / np.maximum(1e-3, lmax - lmin)

    return img_normed

def classifier(img):
    """
    Parameters
    ---------
    img: np.ndarray
        2D array
        value = 0: background
        value > 0: character regions
    Returns
    ---------
    class_index: int
        0-9: numeric symbol
        10: probably other symbol
        11: probably noise
    score: float
        score = max NN output - second NN output
    """

    img = normalize(img)
    x = img[np.newaxis, np.newaxis, :, :]
    y = model(x).data[0]
    
    index_sorted = np.argsort(y)[::-1]
    max_index, second_index = index_sorted[0], index_sorted[1]

    score = y[max_index] - y[second_index]

    return max_index, score


img = cv2.imread('../dataset/test2.png', cv2.IMREAD_GRAYSCALE)
img = 1 - img.astype(np.float32) / 255

img_warped = normalize(img) * 255

cv2.imwrite('../dataset/test_warped.png', img_warped)

class_index, score = classifier(img)

print(class_index)
print(score)