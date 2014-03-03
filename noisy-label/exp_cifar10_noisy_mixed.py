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
noise_sz = int(sys.argv[2])
back_sz = int(sys.argv[3])

# setting
batch_size = 128
param_file = '/home/sainbar/fastnet-confussion-layer/config/cifar-10-18pct-confussion11x22.cfg'
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

# noisy data
noisy_data, noisy_labels = data_loader.load_noisy_labeled()
noisy_data = noisy_data - data_mean
noisy_labels += 11

# background noise
back_data = data_loader.load_noise()
back_data = back_data - data_mean
back_labels = np.ones(back_data.shape[1]) * 10

train_data = np.concatenate((train_data[:,0:pure_sz], noisy_data[:,0:noise_sz], back_data[:,0:back_sz]), axis=1)
train_labels = np.concatenate((train_labels[0:pure_sz], noisy_labels[0:noise_sz], back_labels[0:back_sz]))

# shuffle data
order = range(pure_sz + back_sz + noise_sz)
np.random.shuffle(order)
train_data = train_data[:,order]
train_labels = train_labels[order]

train_batches = data_loader.prepare_batches(train_data, train_labels, batch_size)
test_batches = data_loader.prepare_batches(test_data, test_labels, batch_size)

print '# train:', train_data.shape[1], 'samples', len(train_batches), 'batches'
print '# test:', test_data.shape[1], 'samples', len(test_batches), 'batches'

s = np.zeros((22,11))
s[:11,:11] = np.eye(11)
s[11:,:11] = np.eye(11)
net.layers[-2].weight = data_loader.copy_to_gpu(s)

w = np.eye(22)
w[11:21,11:21] = 0.5 * np.eye(10) + (np.ones((10,10)) - np.eye(10)) * 0.5 / 9.0
net.W_denoise = data_loader.copy_to_gpu(w)
net.label_tmp = data_loader.copy_to_gpu(np.zeros((22,128)))
net.eps1 = 0.01
net.eps2 = float(sys.argv[4])

# net_trainer_noisy.train_mixed(net, 150, train_batches, test_batches)
# net.adjust_learning_rate(0.1)
# net_trainer_noisy.train_mixed(net, 10, train_batches, test_batches)
# net.adjust_learning_rate(0.1)
# net_trainer_noisy.train_mixed(net, 10, train_batches, test_batches)
# save_net_cifar10.save_net(net, 'results/weights-cifar10-pure10k-noisy20k-TD-tc' + str(sys.argv[4]))

