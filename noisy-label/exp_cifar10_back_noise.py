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
num_epoch = 10
learning_rate = 1
image_color = 3
image_size = 32
image_shape = (image_color, image_size, image_size, batch_size)
init_model = parser.parse_config_file(param_file)
net = fastnet.net.FastNet(learning_rate, image_shape, init_model)

# confussion matrix
l = net.layers[-2]
w = np.eye(11)
w[:,10] = 0.05
w[10,10] = 0.5
#w = np.concatenate((w, 0.1*np.ones((10,1))), axis=1)
l.weight = data_loader.copy_to_gpu(w)


# prepare data
data, labels, test_data, test_labels = data_loader.load_cifar10()
data_mean = data.mean(axis=1,keepdims=True)
data = data - data_mean
test_data = test_data - data_mean

# noisy data
noisy_data = data_loader.load_noise()
noisy_data = noisy_data - data_mean
noisy_labels = np.ones(noisy_data.shape[1]) * 10

# mix data
pure_sz = int(sys.argv[1])
noise_sz = int(sys.argv[2])

for i in range(noise_sz/2):
	noisy_labels[i] = np.random.randint(10)

train_data = np.concatenate((data[:,0:pure_sz], noisy_data[:,0:noise_sz]), axis=1)
train_labels = np.concatenate((labels[0:pure_sz], noisy_labels[0:noise_sz]))

order = range(train_data.shape[1])
np.random.shuffle(order)
train_data = train_data[:,order]
train_labels = train_labels[order]

train_batches = data_loader.prepare_batches(train_data, train_labels, batch_size)
test_batches = data_loader.prepare_batches(test_data, test_labels, batch_size)
print 'train:', train_data.shape[1], 'samples', len(train_batches), 'batches'
print 'test:', test_data.shape[1], 'samples', len(test_batches), 'batches'
net_trainer.train(net, num_epoch, train_batches, test_batches)
# net.adjust_learning_rate(0.1)
# net_trainer.train(net, 10, train_batches, test_batches)
# net.adjust_learning_rate(0.1)
# net_trainer.train(net, 10, train_batches, test_batches)
