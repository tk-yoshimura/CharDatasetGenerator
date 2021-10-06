import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

class Model(chainer.Chain):
    def __init__(self):
        super().__init__(
            conv1 = L.Convolution2D(in_channels= 1, out_channels=16, ksize = 3, pad = 1),
            conv2 = L.Convolution2D(in_channels=16, out_channels=64, ksize = 3, pad = 1),
            fc1    = L.Linear(in_size=1024, out_size=32),
            fc2    = L.Linear(in_size=32,   out_size=32),
            fc3    = L.Linear(in_size=32,   out_size=11)
        )

    def __call__(self, x, dropout = False):
        #  1, 16, 16 -> 16, 8, 8
        h1 = F.relu(F.max_pooling_2d(self.conv1( x), ksize=2))
        
        # 16, 8, 8   -> 64, 4, 4
        h2 = F.relu(F.max_pooling_2d(self.conv2(h1), ksize=2))
        
        # dropout
        if dropout:
            h2 = F.dropout(h2, 0.50)

        # 64, 4, 4 -> 1024 -> 32
        h3  = F.relu(self.fc1(h2))

        # dropout
        if dropout:
            h3 = F.dropout(h3, 0.25)

        # 32 -> 32
        h4  = F.relu(self.fc2(h3))

        # dropout
        if dropout:
            h4 = F.dropout(h4, 0.25)

        # 32 -> 11
        y  = self.fc3(h4)

        return y

    def loss(self, x, t, w, dropout = False):
        y = self(x, dropout)

        l = F.softmax_cross_entropy(y, t, class_weight=w)

        acc = F.accuracy(y, t)
        numeric_acc = F.accuracy(y, t, ignore_label=10)

        return l, acc.data, numeric_acc.data

    def save(self, file):
        chainer.serializers.save_npz(file, self)
    
    def load(self, file):
        chainer.serializers.load_npz(file, self)