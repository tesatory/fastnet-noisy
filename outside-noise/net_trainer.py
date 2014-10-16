import fastnet.net
from fastnet.layer import TRAIN, TEST
from fastnet.cuda_kernel import *
from pycuda import gpuarray
import data_loader
import numpy as np
import time
import net_checkpoint
import matplotlib.pylab as plt

def show_stat(net):
    plt.clf()

    f = plt.gcf()
    f.add_subplot('211')
    plt.title(net.checkpoint_name)
    plt.plot(net.stat['epoch'], net.stat['train']['error'], label='train')
    plt.plot(net.stat['epoch'], net.stat['val']['error'], label='val')
    plt.plot(net.stat['epoch'], net.stat['test']['error'], label='test')
    plt.legend(loc = 'lower left')
    plt.ylabel('error')
    plt.xlabel('epochs')
    plt.grid()

    f.add_subplot('212')
    plt.plot(net.stat['epoch'], net.stat['train']['cost'], label='train')
    plt.plot(net.stat['epoch'], net.stat['val']['cost'], label='val')
    plt.plot(net.stat['epoch'], net.stat['test']['cost'], label='test')
    plt.legend(loc = 'lower left')
    plt.ylabel('cost')
    plt.xlabel('epochs')
    plt.grid()

    plt.draw()
    plt.savefig(net.output_dir + 'stat.png')
    time.sleep(0.05)

def train_data(net, batches, stat, train):
    total_cases = 0
    total_cost = 0
    total_correct = 0
    total_error = 0
    N = len(batches)
    order = range(N)
    np.random.shuffle(order)
    for n in xrange(N):
        if train == TRAIN:
            # batch = batches[n]
            batch = batches[order[n]]
            # batch = batches[np.random.randint(len(batches))]
        else:
            batch = batches[n]
        net.train_batch(batch.data, batch.labels, train)
        cost, correct, num_case = net.get_batch_information()
        total_cases += num_case
        total_correct += correct * num_case
        total_cost += cost * num_case
    if total_cases > 0:
        total_error = (1. - 1.0*total_correct/total_cases)
        total_cost = (1.0*total_cost/total_cases)
    if not 'error' in stat:
        stat['error'] = list()
    stat['error'].append(total_error)
    if not 'cost' in stat:
        stat['cost'] = list()
    stat['cost'].append(total_cost)

def train(net, num_epoch, train_batches, val_batches, test_batches = []):
    if hasattr(net, 'stat') == False:
        net.stat = dict()
        net.stat['epoch'] = list()
        net.stat['epsW'] = list()
        net.stat['train'] = dict()
        net.stat['test'] = dict()
        net.stat['val'] = dict()

    fstat = open(net.output_dir + 'stat.txt','aw')

    # disable noise model during testing
    if hasattr(net.layers[-2],'weight'):
        M = net.layers[-2].weight.shape[0]
        w_test = np.eye(M)
        if M == 11:
            w_test[:10,10] = 0.1
            w_test[10,10] = 0
        w_test = data_loader.copy_to_gpu(w_test)


    for n in range(1, num_epoch + 1):
        epoch = len(net.stat['epoch']) + 1
        train_data(net, train_batches, net.stat['train'], TRAIN)
        if hasattr(net.layers[-2],'weight') == False:
            train_data(net, val_batches, net.stat['val'], TEST)
            train_data(net, test_batches, net.stat['test'], TEST)
        else:
            w_noisy = net.layers[-2].weight
            net.layers[-2].weight = w_test
            train_data(net, val_batches, net.stat['val'], TEST)
            train_data(net, test_batches, net.stat['test'], TEST)
            net.layers[-2].weight = w_noisy

        net.stat['epoch'].append(epoch)
        net.stat['epsW'].append(net.layers[1].epsW)
        show_stat(net)

        msg = '%d %0.2e %0.3f %0.3f %0.3f' % (epoch, net.stat['epsW'][-1],
            net.stat['train']['error'][-1], net.stat['val']['error'][-1], 
            net.stat['test']['error'][-1])
        print msg
        fstat.write(msg + '\n')

        if epoch % 10 == 0:
            net_checkpoint.save(net, net.output_dir + 'model_' + str(epoch))

    fstat.close()