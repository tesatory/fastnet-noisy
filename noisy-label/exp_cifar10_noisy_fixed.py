from fastnet import parser
import fastnet.net
import numpy as np
import cudaconv2
import sys
import scipy.io
import net_trainer_noisy
import data_loader

# setting
batch_size = 128
param_file = '/home/sainbar/fastnet-confussion-layer/config/cifar-10-18pct-confussion14.cfg'
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

back_labels[0:40000] = np.random.randint(0, 10, [40000])
noisy_data = np.concatenate((train_data[:,10000:20000], back_data[:,0:50000]), axis=1)
noisy_labels = np.concatenate((train_labels[10000:20000], back_labels[0:50000]))

w = np.zeros([11, 14])
w[:10,:10] = np.eye(10) / 6.0
w[:10,:10] += 4.0 / 60.0
w[10,:10] = 1.0 / 6.0
w[:10,10:] = 4.0 / 50.0
w[10,10:] = 1.0 / 5.0
net.layers[-2].weight = data_loader.copy_to_gpu(w)

# shuffle data
order = range(10000)
np.random.shuffle(order)
train_data = train_data[:,order]
train_labels = train_labels[order]

order = range(60000)
np.random.shuffle(order)
noisy_data = noisy_data[:,order]
noisy_labels = noisy_labels[order]

train_batches = data_loader.prepare_batches(train_data, train_labels, batch_size)
noisy_batches = data_loader.prepare_batches(noisy_data, noisy_labels, batch_size)
test_batches = data_loader.prepare_batches(test_data, test_labels, batch_size)

print 'train:', train_data.shape[1], 'samples', len(train_batches), 'batches'
print 'noisy:', noisy_data.shape[1], 'samples', len(noisy_batches), 'batches'
print 'test:', test_data.shape[1], 'samples', len(test_batches), 'batches'
