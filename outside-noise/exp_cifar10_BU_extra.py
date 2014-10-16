from fastnet import parser
import fastnet.net
import numpy as np
import cudaconv2
import sys
import os
import os.path
import scipy.io
import net_trainer
import data_loader
import net_checkpoint

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('-c', '--clean', default = 50000, type=int)
argparser.add_argument('-n', '--noisy', default = 0, type=int)
argparser.add_argument('-b', '--back', default = 0, type=int)
argparser.add_argument('-m', '--model', default = '')
argparser.add_argument('-e', '--extra', default = 0, type=int)
argparser.add_argument('-w', '--wdecayX', default = 1, type=float)
args = argparser.parse_args()

pure_sz = args.clean
noisy_sz = args.noisy
back_sz = args.back
extra_sz = args.extra

print '# pure', pure_sz, 'noisy_sz', noisy_sz, 'back_sz', back_sz, 'extra_sz', extra_sz

# setting
batch_size = 128
if args.model == 'big1':
	param_file = '../config/cifar-10-18pct-confussion-big1.cfg'
elif args.model == 'big2':
	param_file = '../config/cifar-10-18pct-confussion-big2.cfg'
elif args.model == 'big3':
	param_file = '../config/cifar-10-18pct-confussion-big3.cfg'
else:
	param_file = '../config/cifar-10-18pct-confussion.cfg'
learning_rate = 1
image_color = 3
image_size = 32
image_shape = (image_color, image_size, image_size, batch_size)
init_model = parser.parse_config_file(param_file)
net = fastnet.net.FastNet(learning_rate, image_shape, init_model)
net.checkpoint_name = 'cifar10'
if args.model != '':
	net.checkpoint_name += '_' + args.model	
net.checkpoint_name += '_clean' + str(int(pure_sz/1000)) + 'k'
net.checkpoint_name += '_noisy' + str(int(noisy_sz/1000)) + 'k'
net.checkpoint_name += '_back' + str(int(back_sz/1000)) + 'k'
net.checkpoint_name += '_extra' + str(int(extra_sz/1000)) + 'k'
if args.wdecayX != 1:
	net.checkpoint_name += '_wdX' + str(args.wdecayX)
	for l in net.layers:
		if hasattr(l,'wc'):
			l.wc *= args.wdecayX	
net.output_dir = '~/data/outside-noise-results/results_BU_extra/' + net.checkpoint_name + '/'
if os.path.exists(net.output_dir) == False:
	os.mkdir(net.output_dir)

# prepare data
clean_data, clean_labels, test_data, test_labels = data_loader.load_cifar10()
data_mean = clean_data.mean(axis=1,keepdims=True)
clean_data = clean_data - data_mean
test_data = test_data - data_mean

# background noise
back_data = data_loader.load_noise()
back_data = back_data - data_mean
back_labels = np.ones(back_data.shape[1]) * 10
for i in range(back_sz):
	back_labels[i] = i % 10 # easy to reproduce

# noisy data
noisy_data, noisy_labels = data_loader.load_noisy_labeled()
noisy_data = noisy_data - data_mean

# mix data
train_data = np.concatenate((clean_data[:,0:pure_sz], noisy_data[:,0:noisy_sz], back_data[:,0:back_sz+extra_sz]), axis=1)
train_labels = np.concatenate((clean_labels[0:pure_sz], noisy_labels[0:noisy_sz], back_labels[0:back_sz+extra_sz]))

val_sz = 0
pure_sz2 = pure_sz + int(1. * val_sz * pure_sz/(pure_sz + noisy_sz + back_sz))
noisy_sz2 = noisy_sz + int(1. * val_sz * noisy_sz/(pure_sz + noisy_sz + back_sz))
back_sz2 = back_sz + int(1. * val_sz * back_sz/(pure_sz + noisy_sz + back_sz))
assert pure_sz2 <= clean_data.shape[1]
assert noisy_sz2 <= noisy_data.shape[1]
assert back_sz2 <= back_data.shape[1]
val_data = np.concatenate((clean_data[:,pure_sz:pure_sz2], noisy_data[:,noisy_sz:noisy_sz2], back_data[:,back_sz:back_sz2]), axis=1)
val_labels = np.concatenate((clean_labels[pure_sz:pure_sz2], noisy_labels[noisy_sz:noisy_sz2], back_labels[back_sz:back_sz2]))

# shuffle data
order = range(train_data.shape[1])
np.random.shuffle(order)
train_data = train_data[:,order]
train_labels = train_labels[order]

train_batches = data_loader.prepare_batches(train_data, train_labels, batch_size)
val_batches = data_loader.prepare_batches(val_data, val_labels, batch_size)
test_batches = data_loader.prepare_batches(test_data, test_labels, batch_size)

print '# train:', train_data.shape[1], 'samples', len(train_batches), 'batches'
print '# val:', val_data.shape[1], 'samples', len(val_batches), 'batches'
print '# test:', test_data.shape[1], 'samples', len(test_batches), 'batches'

# confussion matrix
alpha = 1.0 * extra_sz / (back_sz + extra_sz + noisy_sz * 0.7)
w = np.eye(11)
w[:10,10] = (1 - alpha) / 10.
w[10,10] = alpha
net.layers[-2].weight = data_loader.copy_to_gpu(w)

if net_checkpoint.try_load(net) == False:
	# net.adjust_learning_rate(2)
	# net_trainer.train(net, 300, train_batches, val_batches, test_batches)
	# net.adjust_learning_rate(0.1)
	# net_trainer.train(net, 10, train_batches, val_batches, test_batches)
	# net.adjust_learning_rate(0.1)
	# net_trainer.train(net, 10, train_batches, val_batches, test_batches)

	net.adjust_learning_rate(2)
	net_trainer.train(net, 50, train_batches, val_batches, test_batches)
	for i in range(10):
		net_trainer.train(net, 20, train_batches, val_batches, test_batches)
		net.adjust_learning_rate(0.1)
		net_trainer.train(net, 3, train_batches, val_batches, test_batches)
		net.adjust_learning_rate(0.1)
		net_trainer.train(net, 2, train_batches, val_batches, test_batches)
		net.adjust_learning_rate(100)
