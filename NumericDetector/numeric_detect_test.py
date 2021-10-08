import cv2
import numpy as np
from numeric_detector import *
from numeric_classifier import *

imgpath = '../tests/test_map_inv.png'

img = 255 - cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
img_disp = cv2.imread(imgpath)

img_threshold = binarize(img, inversion=False)

cv2.imwrite('../results/test_thres.png', img_threshold)

bounds = list_bounds(img_threshold)

img_bounds = img_disp.copy()
for bound in bounds:
    img_bounds = cv2.rectangle(img_bounds, bound, color=[255, 0, 0], thickness=1)

cv2.imwrite('../results/test_bounds.png', img_bounds)

grouped_bounds = group_bounds(bounds)

img_groupbounds = img_disp.copy()
for j, grouped_bound in enumerate(grouped_bounds):
    for bound in grouped_bound:
        img_groupbounds = cv2.rectangle(img_groupbounds, bound, color=[(j % 4 + 4) * 32, 0, 0], thickness=1)

cv2.imwrite('../results/test_groupbounds.png', img_groupbounds)

grouped_bounds = sift_groupedbounds(grouped_bounds, (4, 32), (6, 48), (0.1, 2))

img_groupbounds = img_disp.copy()
for j, grouped_bound in enumerate(grouped_bounds):
    for bound in grouped_bound:
        img_groupbounds = cv2.rectangle(img_groupbounds, bound, color=[(j % 4 + 4) * 32, 0, 0], thickness=1)

cv2.imwrite('../results/test_siftgroupbounds.png', img_groupbounds)

img_text = img_disp.copy()
for j, grouped_bound in enumerate(grouped_bounds):
    img_tweakbounds = img_disp.copy()

    tweaked_bounds = tweak_bounds(bounds, grouped_bound)
    
    for bound in tweaked_bounds:
        img_tweakbounds = cv2.rectangle(img_tweakbounds, bound, color=[255, 0, 0], thickness=1)

    img_text = cv2.rectangle(img_text, bound_rects(tweaked_bounds), color=[0, 255, 0], thickness=1)

    text, score = parse(img, tweaked_bounds, maxdotsize=5)
    x, y = bound_rects(tweaked_bounds)[:2]
    cv2.putText(img_text, text, (x - 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color=[0, 0, 255], lineType=cv2.LINE_AA)

    cv2.imwrite('../results/test_tweakbounds_{}.png'.format(j), img_tweakbounds)

cv2.imwrite('../results/test_text.png', img_text)