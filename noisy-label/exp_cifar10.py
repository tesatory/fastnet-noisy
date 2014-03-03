from fastnet import parser
import fastnet.net
import numpy as np
import cudaconv2
import sys
import scipy.io
import net_trainer
import data_loader
from pycuda import gpuarray

# setting
batch_size = 128
param_file = '/home/sainbar/fastnet-confussion-layer/config/cifar-10-18pct.cfg'
num_epoch = 60
learning_rate = 1
image_color = 3
image_size = 32
image_shape = (image_color, image_size, image_size, batch_size)
init_model = parser.parse_config_file(param_file)

net = fastnet.net.FastNet(learning_rate, image_shape, init_model)
#l = net.layers[-2];
#l.weight = data_loader.copy_to_gpu(np.eye(10))

# prepare data
train_data, train_labels, test_data, test_labels = data_loader.load_cifar10()
data_mean = train_data.mean(axis=1,keepdims=True)
train_data = train_data - data_mean
test_data = test_data - data_mean

train_batches = data_loader.prepare_batches(train_data, train_labels, batch_size)
test_batches = data_loader.prepare_batches(test_data, test_labels, batch_size)
print 'train:', train_data.shape[1], 'samples', len(train_batches), 'batches'
print 'test:', test_data.shape[1], 'samples', len(test_batches), 'batches'
#net_trainer.train(net, num_epoch, train_batches, test_batches)
#net.adjust_learning_rate(0.1)
#net_trainer.train(net, 10, train_batches, test_batches)
