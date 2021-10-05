import os
import numpy as np
from model import Model 

from chainer import datasets, iterators, optimizers
import chainer.links as L

imgsize = 16

numerics  = np.load('../dataset/numeric_size_{}.npz'.format(imgsize))
alphabets = np.load('../dataset/alphabet_size_{}.npz'.format(imgsize))

batchsize = 1024
iterations = 10000
dirpath_results = 'results/'

os.makedirs(dirpath_results, exist_ok=True)

xs, ts = [], []
class_weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.2 / 44], dtype=np.float32)

for i, c in enumerate('0123456789'):
    x = numerics[c][:, np.newaxis, :, :]
    t = np.full(len(x), i, dtype=np.int32)

    xs.append(x)
    ts.append(t)

for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
    if c in 'gloqzIOZ':
        continue

    x = alphabets[c][:, np.newaxis, :, :]
    t = np.full(len(x), 10, dtype=np.int32)

    xs.append(x)
    ts.append(t)

xs = np.concatenate(xs, axis=0).astype(np.float32) / 255
ts = np.concatenate(ts, axis=0)

dataset = datasets.TupleDataset(xs, ts)
trainset, testset = datasets.split_dataset_random(dataset, len(dataset) * 9 // 10)

train_iter = iterators.SerialIterator(trainset, batchsize, shuffle=True)
test_iter = iterators.SerialIterator(testset, batchsize, shuffle=True)

model = Model()

optimizer = optimizers.Adam(1e-3)
optimizer.setup(model)

with open(dirpath_results + 'loss.csv', 'w') as f:
    f.write('iter,loss,acc,numeric_acc\n')

    for iter in range(iterations + 1):
        batch = train_iter.next()
        x, t = np.stack([item[0] for item in batch], axis=0), np.stack([item[1] for item in batch], axis=0)

        mask = np.where(np.random.uniform(size=x.shape) < 0.3, 1, 0).astype(np.float32)
        r1 = np.random.uniform(size=x.shape).astype(np.float32)
        r2 = np.random.uniform(low=-0.2, high=+0.2, size=x.shape).astype(np.float32)
        x = x * (1 - mask) + r1 * mask + r2

        model.zerograds()
        loss, acc, numeric_acc = model.loss(x, t, class_weights, dropout=True)
        loss.backward()
        optimizer.update()

        print('[Train] iter: %d loss: %.4f acc: %.4f num_acc: %.4f' % (iter, loss.data, acc, numeric_acc))

        if iter > 0 and iter % 100 == 0:
            batch = test_iter.next()
            x, t = np.stack([item[0] for item in batch], axis=0), np.stack([item[1] for item in batch], axis=0)

            loss, acc, numeric_acc = model.loss(x, t, class_weights, dropout=False)

            print('[Test]  iter: %d loss: %.4f acc: %.4f num_acc: %.4f' % (iter, loss.data, acc, numeric_acc))
            f.write('%d,%f,%f,%f\n' % (iter, loss.data, acc, numeric_acc))
            f.flush()

            model.save(dirpath_results + 'model_snap_%d.npz' % iter)
            optimizer.alpha *= 0.95

model.save('../model/numeric_classifier_model.npz')