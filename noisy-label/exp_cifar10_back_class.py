from fastnet import parser
import fastnet.net
import numpy as np
import cudaconv2
import sys
import scipy.io
import net_trainer_noisy
import data_loader
import confusion_matrix
import save_net_cifar10

pure_sz = int(sys.argv[1])
back_sz = int(sys.argv[2])

# setting
batch_size = 128
param_file = '/home/sainbar/fastnet-confussion-layer/config/cifar-10-18pct-confussion.cfg'
learning_rate = 1
image_color = 3
image_size = 32
image_shape = (image_color, image_size, image_size, batch_size)
init_model = parser.parse_config_file(param_file)
net = fastnet.net.FastNet(learning_rate, image_shape, init_model)

# prepare data
train_data, train_labels, test_data, test_labels = data_loader.load_cifar10()
data_mean = train_data.mean(axis=1,keepdims=True)
train_data = train_data - data_mean
test_data = test_data - data_mean

# background noise
back_data = data_loader.load_noise()
back_data = back_data - data_mean
back_labels = np.ones(back_data.shape[1]) * 10

# confussion matrix
w = np.eye(11)
net.layers[-2].weight = data_loader.copy_to_gpu(w)

# shuffle data
order = range(pure_sz)
np.random.shuffle(order)
train_data = train_data[:,order]
train_labels = train_labels[order]
order = range(back_sz)
np.random.shuffle(order)
back_data2 = back_data[:,back_sz:back_sz+10000]
back_labels2 = back_labels[back_sz:back_sz+10000]
back_data = back_data[:,order]
back_labels = back_labels[order]

train_batches = data_loader.prepare_batches(train_data, train_labels, batch_size)
back_batches = data_loader.prepare_batches(back_data, back_labels, batch_size)
back_batches2 = data_loader.prepare_batches(back_data2, back_labels2, batch_size)
test_batches = data_loader.prepare_batches(test_data, test_labels, batch_size)

print 'train:', train_data.shape[1], 'samples', len(train_batches), 'batches'
print 'back:', back_data.shape[1], 'samples', len(back_batches), 'batches'
print 'test:', test_data.shape[1], 'samples', len(test_batches), 'batches'
