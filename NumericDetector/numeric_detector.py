import cv2
import numpy as np

def binarize(img: np.ndarray, inversion=False):
    """
    Parameters
    ---------
    img: np.ndarray
        2D uint8 array
    inversion: bool
        False : black-background white-objects
        True  : white-background black-objects
    Returns
    ---------
    img_binary: np.ndarray
        2D uint8 array
    """

    if img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError('img')

    if inversion:
        return cv2.adaptiveThreshold(img, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 
                                     blockSize=9, C=8)
    else:
        return cv2.adaptiveThreshold(img, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 
                                     blockSize=9, C=-8)

def list_bounds(img_binary: np.ndarray):
    """
    Parameters
    ---------
    img_binary: np.ndarray
        2D uint8 array
        black-background white-objects
    Returns
    ---------
    bounds: np.ndarray
        N x 4 2D array
        bound: (left, top, width, height)
    """

    if img_binary.ndim != 2 or img_binary.dtype != np.uint8:
        raise ValueError('img_binary')

    contours, _ = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    bounds = []
    for contour in contours:
        area = cv2.contourArea(contour, oriented=True)
        if area < 0:
            bounds.append(cv2.boundingRect(contour))

    bounds = np.stack(bounds)

    return bounds

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

def is_inside(bound_inner: np.ndarray, bound_outer: np.ndarray) -> bool:
    """
    Parameters
    ---------
    bound_inner: np.ndarray
        1D array
        (left, top, width, height)
    bound_outer: np.ndarray
        1D array
        (left, top, width, height)
    Returns
    ---------
    inside: bool
    """

    if bound_inner.shape != (4,):
        raise ValueError('bound1')
    if bound_outer.shape != (4,):
        raise ValueError('bound2')

    coord_inner, size_inner = bound_inner[:2], bound_inner[2:]
    coord_outer, size_outer = bound_outer[:2], bound_outer[2:]
 
    if np.any(coord_inner < coord_outer):
        return False

    if np.any(coord_inner - coord_outer + size_inner > size_outer):
        return False
    
    return True

def sift_bounds(bounds: np.ndarray, x_range: tuple, y_range: tuple, w_range: tuple, h_range: tuple, mode: str, additional_cond = None) -> int:
    """
    Parameters
    ---------
    bounds: np.ndarray
        N x 4 2D array
        bound: (left, top, width, height)
    x_range: (int, int)
        left bbox coord range
    y_range: (int, int)
        top, bottom coord range
    w_range: (int, int)
    h_range: (int, int)
    mode: str
        'most_left', 'most_right'
    additional_cond: np.ndarray or None
        N 1D boolean array
    Returns
    ---------
    bound_index: inr or None
    """
    
    if bounds.ndim != 2 or bounds.shape[1] != 4:
        raise ValueError('bounds')
    if not mode in ['most_left', 'most_right']:
        raise ValueError('mode')
    if not additional_cond is None \
        and additional_cond.dtype != np.bool \
        and additional_cond.shape != (bounds.shape[0],):

        raise ValueError('additional_cond')

    x_min, x_max = x_range
    y_min, y_max = y_range
    w_min, w_max = w_range
    h_min, h_max = h_range

    ls, ts, ws, hs = bounds[:, 0], bounds[:, 1], bounds[:, 2], bounds[:, 3]
    bs = ts + hs

    inbox = np.logical_and(
        np.logical_and(np.logical_and(ls >= x_min, ls <= x_max), np.logical_and(ts >= y_min, bs <= y_max)),
        np.logical_and(np.logical_and(ws >= w_min, ws <= w_max), np.logical_and(hs >= h_min, hs <= h_max)))

    if additional_cond is None:
        candidate_indexes = np.where(inbox)[0]
    else:
        candidate_indexes = np.where(np.logical_and(inbox, additional_cond))[0]

    if len(candidate_indexes) < 1:
        return None

    candidate_bounds = bounds[candidate_indexes]
    
    if mode is 'most_left':
        bound_index = candidate_indexes[
                np.lexsort((
                    candidate_bounds[:, 2] * candidate_bounds[:, 3],
                    -candidate_bounds[:, 0]))[-1]]
    else:
        bound_index = candidate_indexes[
                np.lexsort((
                    candidate_bounds[:, 2] * candidate_bounds[:, 3],
                    candidate_bounds[:, 0] + candidate_bounds[:, 2]))[-1]]

    return bound_index

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

    group_index = 0

    for i in range(n):
        if g[i] >= 0:
            continue

        g[i] = group_index
        groups = [bounds[i],]

        prev_bound_index = i
        while prev_bound_index < n:
            l, t, w, h = bounds[prev_bound_index]
            
            x_min, x_max = l + w - 1, l + w * 2
            y_min, y_max = t - max(1, h // 4), t + h + max(1, h // 4)
            w_min, w_max = w // 4, w * 4
            h_min, h_max = h * 3 // 4, h * 5 // 4

            bound_index = sift_bounds(bounds, (x_min, x_max), (y_min, y_max), (w_min, w_max), (h_min, h_max), 
                                      mode = 'most_left', additional_cond = g<0)

            if bound_index is None:
                break

            g[bound_index] = group_index
            group_index += 1
            prev_bound_index = bound_index
            groups.append(bounds[bound_index])

        groups = np.stack(groups, axis=0)
        list_bounds.append(groups)

    return list_bounds

def sift_groupedbounds(list_bounds: list, w_range: tuple, h_range: tuple, aspect_range = None) -> list:
    """
    Parameters
    ---------
    list_bounds: list of np.ndarray
        list of N x 4 2D array
        bound: (left, top, width, height)
    w_range: (int, int)
        range of max width bounds 
    h_range: (int, int)
        range of max height bounds
    aspect_range: (float, float) or None
        range of median aspect(width/height) bounds
    Returns
    ---------
    list_bounds: list of np.ndarray
        list of N x 4 2D array
        bound: (left, top, width, height)
    """

    if type(list_bounds) != list:
        raise ValueError('list_bounds')

    w_min, w_max = w_range
    h_min, h_max = h_range
    aspect_min, aspect_max = aspect_range if aspect_range is not None else (None, None)

    new_list_bounds = []

    for bounds in list_bounds:
        if bounds.ndim != 2 or bounds.shape[1] != 4:
            raise ValueError('bounds')

        w, h = np.max(bounds[:, 2]), np.min(bounds[:, 3])

        if w < w_min or w > w_max or h < h_min or h > h_max:
            continue

        if aspect_range is not None:
            aspects = w / np.maximum(1e-5, h)
            aspect_median = np.median(aspects)

            if aspect_median < aspect_min or aspect_median > aspect_max:
                continue

        new_list_bounds.append(bounds)

    return new_list_bounds

def tweak_bounds(bounds: np.ndarray, grouped_bounds: np.ndarray, detect_dot = True, detect_sign = True) -> np.ndarray:
    """
    Parameters
    ---------
    bounds: np.ndarray
        N x 4 2D integer array
        bound: (left, top, width, height)
    grouped_bounds: np.ndarray
        N x 4 2D integer array
        bound: (left, top, width, height)
    detect_sign: bool
    Returns
    ---------
    grouped_bounds: np.ndarray
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

    if detect_sign:
        l, t, _, h = grouped_bounds[0]
            
        x_min, x_max = l - width_median * 3 // 2, l - width_median // 2
        y_min, y_max = t, t + h
        w_min, w_max = width_median // 2, width_median * 2
        h_min, h_max = h // 16, h * 2 // 3

        bound_index = sift_bounds(bounds, (x_min, x_max), (y_min, y_max), (w_min, w_max), (h_min, h_max), 
                                  mode = 'most_right')

        if not bound_index is None:
            new_bounds.append(bounds[bound_index])
            
    for i, (w, h, bound) in enumerate(zip(widths, heights, grouped_bounds)):
        n = int(np.floor(w / width_median + 0.2))
    
        if n <= 1 or w / h < 0.9:
            new_bounds.append(bound)
        else:
            bx, by, bw, bh = bound
            for j in range(n):
                new_bounds.append(np.array([bx + bw * j // n, by, bw // n, bh]))
        
        if detect_dot and i < len(grouped_bounds) - 1:
            bx, by, bw, bh = bound
            
            x_min, x_max = bx + bw - 1, bx + 2 * bw
            y_min, y_max = by + max(1, bh // 2), by + bh + max(1, bh // 8)
            w_min, w_max = bw // 16, bw
            h_min, h_max = bh // 16, bh // 2
        
            bound_index = sift_bounds(bounds, (x_min, x_max), (y_min, y_max), (w_min, w_max), (h_min, h_max), 
                                      mode = 'most_left')
        
            if bound_index is None:
                continue

            bound_dot, bound_next = bounds[bound_index], grouped_bounds[i + 1]

            if is_inside(bound_dot, bound_next):
                continue

            new_bounds.append(bound_dot)
                        
    grouped_bounds = np.stack(new_bounds) if len(new_bounds) > 0 else np.zeros((0, 4), np.int)

    return grouped_bounds

def bound_rects(rects: np.ndarray) -> np.ndarray:
    """
    Parameters
    ---------
    rects: np.ndarray
        N x 4 2D integer array
        rect: (left, top, width, height)
    Returns
    ---------
    bound_rects: np.ndarray
        1D array
        (left, top, width, height)
    """

    if rects.ndim != 2 or rects.shape[1] != 4:
        raise ValueError('rects')

    left, top = rects[:, 0], rects[:, 1]
    right, bottom = left + rects[:, 2], top + rects[:, 3]

    left_min, top_min = np.min(left), np.min(top)
    right_max, bottom_max = np.max(right), np.max(bottom)

    bound_rects = np.array([left_min, top_min, right_max - left_min, bottom_max - top_min], dtype=rects.dtype)

    return bound_rects