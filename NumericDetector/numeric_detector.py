import cv2
import numpy as np

def compute_iou(bound1: np.ndarray, bound2: np.ndarray) -> float:
    """
    Parameters
    ---------
    bound1: np.ndarray
        1D array
        (left, top, width, height)
    bound2: np.ndarray
        1D array
        (left, top, width, height)
    Returns
    ---------
    iou: float
    """

    if bound1.shape != (4,):
        raise ValueError('bound1')
    if bound2.shape != (4,):
        raise ValueError('bound2')

    rect1_area, rect2_area = bound1[2] * bound1[3], bound2[2] * bound2[3]

    rect1 = np.concatenate([bound1[:2], bound1[:2] + bound1[2:]], axis=0)
    rect2 = np.concatenate([bound2[:2], bound2[:2] + bound2[2:]], axis=0)

    size  = np.maximum(0, np.minimum(rect1[2:], rect2[2:]) - np.maximum(rect1[:2], rect2[:2]))
    intersect_area = size[0] * size[1]

    iou = intersect_area / np.maximum(1e-5, rect1_area + rect2_area - intersect_area)

    return iou


def group_bounds(bounds: np.ndarray, concat_iou_threshold = 0.05) -> list:
    """
    Parameters
    ---------
    bounds: np.ndarray
        N x 4 2D array
        bound: (left, top, width, height)
    concat_iou_threshold: float
        minimum iou that concat bound
    Returns
    ---------
    list_bounds: list of np.ndarray
        list of N x 4 2D array
        bound: (left, top, width, height)
    """

    if bounds.ndim != 2 or bounds.shape[1] != 4:
        raise ValueError('bounds')

    n = bounds.shape[0]
    g = np.full(n, -1, dtype=np.int)
    list_bounds = []

    bounds = bounds[np.argsort(bounds[:, 0])]

    group_index = 0

    for i in range(n):
        if g[i] >= 0:
            continue

        g[i] = group_index
        groups = [bounds[i],]

        j = i

        while j < n:
            bound = bounds[j]
            expected_bound = np.concatenate([bound[:1] + bound[2], bound[1:2], bound[2:]], axis=0)

            max_iou = concat_iou_threshold
            max_index = -1

            for k in range(j + 1, n):
                if g[k] >= 0:
                    continue

                iou = compute_iou(expected_bound, bounds[k])
                if iou > max_iou:
                    max_iou = iou
                    max_index = k

            if max_index >= 0:
                g[max_index] = group_index
                group_index += 1
                j = max_index
                groups.append(bounds[max_index])
            else:
                break

        groups = np.stack(groups, axis=0)
        list_bounds.append(groups)

    return list_bounds

img = cv2.imread('../dataset/test_map.png', cv2.IMREAD_GRAYSCALE)
img_disp = cv2.imread('../dataset/test_map.png')

img_mask = np.where(img > 96, 1, 0).astype(np.uint8)
img_threshold = img_mask * cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)

cv2.imwrite('../dataset/test_thres.png', img_threshold)

contours, _ = cv2.findContours(img_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

bounds = np.stack([cv2.boundingRect(c) for c in contours])
bounds = bounds[bounds[:, 3] > 5]

img_bounds = img_disp.copy()
for bound in bounds:
    img_bounds = cv2.rectangle(img_bounds, bound, color=[255, 0, 0], thickness=1)

cv2.imwrite('../dataset/test_bounds.png', img_bounds)

groups = group_bounds(bounds)

img_gounpbounds = img_disp.copy()
for i, group in enumerate(groups):
    if len(group) <= 1:
        continue

    for bound in group:
        img_gounpbounds = cv2.rectangle(img_gounpbounds, bound, color=[((i + 1) % 8) * 32, 0, 0], thickness=1)

cv2.imwrite('../dataset/test_gounpbounds.png', img_gounpbounds)