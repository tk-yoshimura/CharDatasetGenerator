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


def group_bounds(bounds: np.ndarray) -> list:
    """
    Parameters
    ---------
    bounds: np.ndarray
        N x 4 2D array
        bound: (left, top, width, height)
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

    ls, ts, bs = bounds[:, 0], bounds[:, 1], bounds[:, 1] + bounds[:, 3]
    cys, sizes = bounds[:, 1] + bounds[:, 3] / 2, bounds[:, 2] * bounds[:, 3]

    group_index = 0

    for i in range(n):
        if g[i] >= 0:
            continue

        g[i] = group_index
        groups = [bounds[i],]

        prev_bound_index = i
        while prev_bound_index < n:
            bound, cy = bounds[prev_bound_index], cys[prev_bound_index]
            top, bottom = ts[prev_bound_index], bs[prev_bound_index]
            
            minx = bound[0] + bound[2] - 1
            maxx = minx + np.minimum(bound[2], bound[3])
            miny = cy - np.maximum(4, bound[3]) / 4
            maxy = cy + np.maximum(4, bound[3]) / 4
            minsize = bound[2] * bound[3] / 8
            maxsize = bound[2] * bound[3] * 4

            candidate_indexes = np.where(
                np.logical_and(
                    np.logical_and(
                        g < 0,
                        np.logical_and(sizes >= minsize, sizes <= maxsize)
                    ),
                    np.logical_and(
                        np.logical_and(ls >= minx, ls <= maxx),
                        np.logical_or(
                            np.logical_and(cys >= miny, cys <= maxy),
                            np.logical_and(top >= ts,   bottom <= bs),
                        )
                    )
                )
            )[0]

            if len(candidate_indexes) < 1:
                break

            candidate_bounds = bounds[candidate_indexes]
            most_large_index = candidate_indexes[
                    np.lexsort((
                        candidate_bounds[:, 2] * candidate_bounds[:, 3],
                        -candidate_bounds[:, 0])
                )[-1]]

            g[most_large_index] = group_index
            group_index += 1
            prev_bound_index = most_large_index
            groups.append(bounds[most_large_index])

        groups = np.stack(groups, axis=0)
        list_bounds.append(groups)

    return list_bounds

def tweak_bounds(bounds: np.ndarray, grouped_bounds: np.ndarray, detect_dot = True, detect_prefix = True) -> np.ndarray:
    """
    Parameters
    ---------
    bounds: np.ndarray
        N x 4 2D integer array
        bound: (left, top, width, height)
    grouped_bounds: np.ndarray
        N x 4 2D integer array
        bound: (left, top, width, height)
    detect_prefix: bool
    Returns
    ---------
    bounds: np.ndarray
        N x 4 2D array
        bound: (left, top, width, height)
    """

    if bounds.ndim != 2 or bounds.shape[1] != 4 or \
        (bounds.dtype != np.int32 and bounds.dtype != np.int64):
        raise ValueError('bounds')

    if grouped_bounds.ndim != 2 or grouped_bounds.shape[1] != 4 or \
        (grouped_bounds.dtype != np.int32 and grouped_bounds.dtype != np.int64):
        raise ValueError('grouped_bounds')

    if len(grouped_bounds) < 1:
        return grouped_bounds

    grouped_bounds = grouped_bounds[np.argsort(grouped_bounds[:, 0])]

    widths, heights = grouped_bounds[:, 2], grouped_bounds[:, 3]
    width_median = np.maximum(0.1, np.median(widths))

    new_bounds = []
    
    ls, rs = bounds[:, 0], bounds[:, 0] + bounds[:, 2]
    cys, sizes = bounds[:, 1] + bounds[:, 3] / 2, bounds[:, 2] * bounds[:, 3]

    if detect_prefix:
        bound = grouped_bounds[0]
            
        cy = bound[1] + bound[3] / 2
        minx = bound[0] - np.minimum(bound[2], bound[3])
        maxx = bound[0] 
        miny = cy - np.maximum(4, bound[3]) / 4
        maxy = cy + np.maximum(4, bound[3]) / 4
        minsize = bound[2] * bound[3] / 16
        maxsize = bound[2] * bound[3]        

        candidate_indexes = np.where(
            np.logical_and(
                np.logical_and(sizes >= minsize, sizes <= maxsize),
                np.logical_and(
                    np.logical_and(rs >= minx, rs <= maxx),
                    np.logical_and(cys >= miny, cys <= maxy)
                )
            )
        )[0]

        if len(candidate_indexes) >= 1:        
            candidate_bounds = bounds[candidate_indexes]
            most_large_index = candidate_indexes[
                np.lexsort((
                    candidate_bounds[:, 2] * candidate_bounds[:, 3],
                    candidate_bounds[:, 0] + candidate_bounds[:, 2])
            )[-1]]

            new_bounds.append(bounds[most_large_index])
            
    for w, h, bound in zip(widths, heights, grouped_bounds):
        n = int(np.floor(w / width_median + 0.2))
    
        if n <= 1 or w / h < 0.9:
            new_bounds.append(bound)
        else:
            bx, by, bw, bh = bound
            for i in range(n):
                new_bounds.append(np.array([bx + bw * i / n, by, bw / n, bh]))

        if detect_dot:
            minx = bound[0] + bound[2] * 4 // 5
            maxx = bound[0] + bound[2] * 2 
            miny = bound[1] + bound[3] * 3 // 4
            maxy = bound[1] + bound[3] + 1
            minsize = bound[2] * bound[3] / 128
            maxsize = bound[2] * bound[3] / 6  

            candidate_indexes = np.where(
                np.logical_and(
                    np.logical_and(sizes >= minsize, sizes <= maxsize),
                    np.logical_and(
                        np.logical_and(ls >= minx, ls <= maxx),
                        np.logical_and(cys >= miny, cys <= maxy)
                    )
                )
            )[0]

            if len(candidate_indexes) >= 1:        
                candidate_bounds = bounds[candidate_indexes]
                most_large_index = candidate_indexes[
                    np.lexsort((
                        candidate_bounds[:, 2] * candidate_bounds[:, 3],
                        -candidate_bounds[:, 0])
                )[-1]]

                new_bounds.append(bounds[most_large_index])
                        
    grouped_bounds = np.stack(new_bounds) if len(new_bounds) > 0 else np.zeros((0, 4), np.int)

    return grouped_bounds


img = cv2.imread('../dataset/test02.png', cv2.IMREAD_GRAYSCALE)
img_disp = cv2.imread('../dataset/test02.png')

img_mask = np.where(img > 96, 1, 0).astype(np.uint8)
img_threshold = img_mask * cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)

cv2.imwrite('../dataset/test_thres.png', img_threshold)

contours, _ = cv2.findContours(img_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

bounds = np.stack([cv2.boundingRect(c) for c in contours])
bounds = bounds[bounds[:, 2] * bounds[:, 3] >= 4]

img_bounds = img_disp.copy()
for bound in bounds:
    img_bounds = cv2.rectangle(img_bounds, bound, color=[255, 0, 0], thickness=1)

cv2.imwrite('../dataset/test_bounds.png', img_bounds)

grouped_bounds = group_bounds(bounds)

img_groupbounds = img_disp.copy()
for i, grouped_bound in enumerate(grouped_bounds):
    if len(grouped_bound) <= 1:
        continue

    for bound in grouped_bound:
        img_groupbounds = cv2.rectangle(img_groupbounds, bound, color=[((i + 1) % 7) * 32, 0, 0], thickness=1)

cv2.imwrite('../dataset/test_groupbounds.png', img_groupbounds)

img_tweakbounds = img_disp.copy()
for i, grouped_bound in enumerate(grouped_bounds):
    if len(grouped_bound) <= 2:
        continue

    for bound in tweak_bounds(bounds, grouped_bound):
        img_tweakbounds = cv2.rectangle(img_tweakbounds, bound, color=[((i + 1) % 7) * 32, 0, 0], thickness=1)

cv2.imwrite('../dataset/test_tweakbounds.png', img_tweakbounds)