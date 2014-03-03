from fastnet import parser
import fastnet.net
import numpy as np
import cudaconv2
import sys
import scipy.io
import net_trainer
import data_loader

# setting
batch_size = 128
param_file = '/home/sainbar/fastnet-confussion-layer/config/cifar-10-18pct-confussion.cfg'
#param_file = '/home/sainbar/fastnet/config/cifar-10-18pct-big3.cfg'
num_epoch = 60
learning_rate = 1
image_color = 3
image_size = 32
image_shape = (image_color, image_size, image_size, batch_size)
init_model = parser.parse_config_file(param_file)
net = fastnet.net.FastNet(learning_rate, image_shape, init_model)

# confussion matrix
l = net.layers[-2];
w = np.eye(10) * 0.8
w[0,:] = w[0,:] + 0.2
l.weight = data_loader.copy_to_gpu(w)

# prepare data
train_data, train_labels, test_data, test_labels = data_loader.load_cifar10()
data_mean = train_data.mean(axis=1,keepdims=True)
train_data = train_data - data_mean
test_data = test_data - data_mean

# flip labels
N = 50000
p = 0.2
for i in range(N):
	if np.random.rand() < p:
		train_labels[i] = 0

order = range(train_data.shape[1])
np.random.shuffle(order)
train_data = train_data[:,order]
train_labels = train_labels[order]

train_batches = data_loader.prepare_batches(train_data, train_labels, batch_size)
test_batches = data_loader.prepare_batches(test_data, test_labels, batch_size)
print 'train:', train_data.shape[1], 'samples', len(train_batches), 'batches'
print 'test:', test_data.shape[1], 'samples', len(test_batches), 'batches'
net_trainer.train(net, num_epoch, train_batches, test_batches)
net.adjust_learning_rate(0.1)
net_trainer.train(net, 10, train_batches, test_batches)
# net.adjust_learning_rate(0.1)
# net_trainer.train(net, 10, train_batches, test_batches)
