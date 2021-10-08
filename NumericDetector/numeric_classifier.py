import cv2
import numpy as np

import chainer
from numeric_classifier_model import Model

imgsize = 16

model = Model()
model.load('../model/numeric_classifier_model.npz')

def normalize(img: np.ndarray) -> np.ndarray:
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

def classifier(img: np.ndarray) -> tuple:
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
    score: float
        score = max NN output - second NN output
    """

    img = normalize(img)
    x = img[np.newaxis, np.newaxis, :, :]

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y = model(x).data[0]
    
    index_sorted = np.argsort(y)[::-1]
    max_index, second_index = index_sorted[0], index_sorted[1]

    score = y[max_index] - y[second_index]

    return max_index, score

def parse(img: np.ndarray, rects: np.ndarray, maxdotsize: int, score_threshold=0.1) -> tuple:
    """
    Parameters
    ---------
    img: np.ndarray
        2D uint8 array
        value = 0: background
        value > 0: character regions
    rects: np.ndarray
        N x 4 2D array
        rect: (left, top, width, height)
    maxdotsize: int
    score_threshold: float
        score threshold that regard as not numeric symbol
    Returns
    ---------
    numeric_text: str
    score: float
    """

    if img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError('img')
    if rects.ndim != 2 or rects.shape[1] != 4 or \
        (rects.dtype != np.int32 and rects.dtype != np.int64):
        raise ValueError('rects')

    text = ''
    scores = []

    rects[:2] = np.maximum(0, rects[:2])
    rects[2:] = np.maximum(1, rects[2:])

    for i, rect in enumerate(rects):
        l, t, w, h = rect

        img_clip = img[t:t+h, l:l+w]
        img_clip_mean = np.mean(img_clip > 196)

        if i == 0 and w > h and img_clip_mean > 0.6:
            text += '-'
            continue
        if w <= h * 3 // 2 and h <= w * 3 // 2 \
            and w <= maxdotsize and h <= maxdotsize \
            and not '.' in text:

            text += '.'
            continue

        class_index, score = classifier(img_clip)
        c = str(class_index) if (class_index < 10 and score >= score_threshold) else '?'
        
        text += c
        scores.append(score if c != '?' else 0)

    score = np.min(np.stack(scores)) if len(scores) > 0 else 0

    return text, score
