import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

class Model(chainer.Chain):
    def __init__(self):
        super().__init__(
            conv1 = L.Convolution2D(in_channels= 1, out_channels= 8, ksize = 3, pad = 1),
            conv2 = L.Convolution2D(in_channels= 8, out_channels=32, ksize = 3, pad = 1),
            fc1    = L.Linear(in_size=512, out_size=32),
            fc2    = L.Linear(in_size=32,  out_size=12)
        )

    def __call__(self, x, dropout = False):
        #  1, 16, 16 ->  8, 8, 8
        h1 = F.relu(F.max_pooling_2d(self.conv1( x), ksize=2))
        
        #  8, 8, 8   -> 32, 4, 4
        h2 = F.relu(F.max_pooling_2d(self.conv2(h1), ksize=2))
        
        # dropout
        if dropout:
            h2 = F.dropout(h2, 0.50)

        # 32, 4, 4 -> 512 -> 32
        h3  = F.relu(self.fc1(h2))

        # dropout
        if dropout:
            h3 = F.dropout(h3, 0.25)

        # 32 -> 12
        y  = self.fc2(h3)

        return y

    def loss(self, x, t, dropout = False):
        y = self(x, dropout)

        l = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)

        return l, acc.data

    def save(self, file):
        chainer.serializers.save_npz(file, self)
    
    def load(self, file):
        chainer.serializers.load_npz(file, self)