import numpy as np

def noise(n: int, imgsize: int):
    x = np.random.uniform(64, 255, (n, imgsize, imgsize)).astype(np.uint8)
    s = n // 10

    for i in range(1, 5):
        c = i

        x[i * s:(i + 1) * s,   :c, :] = 0
        x[i * s:(i + 1) * s, -c:,  :] = 0

    for i in range(5, 10):
        c = i - 5

        x[i * s:(i + 1) * s, :,   :c] = 0
        x[i * s:(i + 1) * s, :, -c: ] = 0

    return x