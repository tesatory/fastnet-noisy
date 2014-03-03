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
net.layers[-2].weight = data_loader.copy_to_gpu(np.eye(11))

# prepare data
train_data, train_labels, test_data, test_labels = data_loader.load_cifar10()
data_mean = train_data.mean(axis=1,keepdims=True)
train_data = train_data - data_mean
test_data = test_data - data_mean

# background noise
back_data = data_loader.load_noise()
back_data = back_data - data_mean
back_labels = np.ones(back_data.shape[1]) * 10

train_data = np.concatenate((train_data[:,0:pure_sz], back_data[:,0:back_sz]), axis=1)
train_labels = np.concatenate((train_labels[0:pure_sz], back_labels[0:back_sz]))
test_data = np.concatenate((test_data, back_data[:,back_sz:back_sz+1000]), axis=1)
test_labels = np.concatenate((test_labels, back_labels[back_sz:back_sz+1000]))
order = range(pure_sz + back_sz)
np.random.shuffle(order)
train_data = train_data[:,order]
train_labels = train_labels[order]

# add noise to label
W = np.load('mixing-matrix-back-' + sys.argv[3] + '.npy')
train_labels_noisy = confusion_matrix.mix_labels(W, train_labels)

train_batches = data_loader.prepare_batches(train_data, train_labels_noisy, batch_size)
train_batches2 = data_loader.prepare_batches(train_data, train_labels, batch_size)
test_batches = data_loader.prepare_batches(test_data, test_labels, batch_size)

print 'train:', train_data.shape[1], 'samples', len(train_batches), 'batches'
print 'test:', test_data.shape[1], 'samples', len(test_batches), 'batches'
