import chainer
import chainer.initializers as I
import chainer.functions as F
import chainer.links as L


class Alex(chainer.Chain):

    """Model definition file for AlexNet 
         without partition toward the channel axis. 
           Based on Chainer sample code for Single-GPU AlexNet.
       Copyright (C) 2016-18 Takashi Shinozaki"""

    def __init__(self):
        # Default initializer is HeNormal. Here, we use standard Normal.
        Norm_01 = I.Normal(scale=0.01)
        Norm_005 = I.Normal(scale=0.005)
        
        super(Alex, self).__init__(
            conv1 = L.Convolution2D(3,  96, 22, stride=8, 
                                    initialW=Norm_01, initial_bias=0.0),
            conv2 = L.Convolution2D(96, 256,  5, pad=2,
                                    initialW=Norm_01, initial_bias=0.1),
            conv3 = L.Convolution2D(256, 384,  3, pad=1,
                                    initialW=Norm_01, initial_bias=0.0),
            conv4 = L.Convolution2D(384, 384,  3, pad=1,
                                    initialW=Norm_01, initial_bias=0.1),
            conv5 = L.Convolution2D(384, 256,  3, pad=1,
                                    initialW=Norm_01, initial_bias=0.1),
            fc6 = L.Linear(6*6*256, 4096, initialW=Norm_005, initial_bias=0.1),
            fc7 = L.Linear(4096, 4096, initialW=Norm_005, initial_bias=0.1),
            fc8 = L.Linear(4096, 1000, initialW=Norm_01, initial_bias=0.0),
        )
        self.insize = 227 * 2 

    def __call__(self, x):
        h = self.conv1(x)
        h = F.local_response_normalization(F.relu(h))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.conv2(h)
        h = F.local_response_normalization(F.relu(h))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.conv3(h)
        h = F.relu(h)
        h = self.conv4(h)
        h = F.relu(h)
        h = self.conv5(h)
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)
        return h

#EOF
