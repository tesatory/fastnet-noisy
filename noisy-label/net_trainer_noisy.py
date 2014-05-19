import fastnet.net
from fastnet.layer import TRAIN, TEST
from fastnet.cuda_kernel import *
from pycuda import gpuarray
import data_loader
import numpy as np
import confusion_matrix
import time
# import matplotlib.pylab as plt

def prob_project(x):
    x = x.copy()
    pos = np.array([True] * x.shape[0])
    eps = 0.0001
    while abs(x.sum() - 1) > eps:
        r = x.sum() - 1
        x[pos] -= r / pos.sum()
        pos = x > 0
        x[~pos] = 0
    return x

def normalize_conf_matrix(net):
    l = net.layers[-2];
    w = l.weight.get()
    for i in range(w.shape[1]):
        w[:,i] = prob_project(w[:,i])
    l.weight.set(w)

def show_stat(net, test_batches):
    plt.clf()
    f = plt.gcf()
    c = confusion_matrix.get_confusion(net, test_batches)
    f.add_subplot('221')
    plt.imshow(c, interpolation = 'nearest')
    plt.colorbar()

    f.add_subplot('222')
    m = data_loader.copy_to_cpu(net.layers[-2].weight)
    plt.imshow(m, interpolation = 'nearest')
    plt.colorbar()

    f.add_subplot('223')
    plt.plot(net.stat['test-error'])

    plt.draw()
    time.sleep(0.05)

def train(net, num_epoch, train_batches, noisy_batches, test_batches, lrate_beta):
    l = net.layers[-2]
    M = l.weight.shape[0]
    assert M == l.weight.shape[1]
    w_pure = data_loader.copy_to_gpu(np.eye(M))
    w_test = np.eye(M)
    if M == 11:
        w_test[:10,10] = 0.1
        w_test[10,10] = 0
    w_test = data_loader.copy_to_gpu(w_test)

    if hasattr(net, 'stat') == False:
        net.stat = dict()

    for epoch in range(num_epoch):
        train_cases = 0
        train_cost = 0
        train_correct = 0
        train_error = 0
        test_cases = 0
        test_cost = 0
        test_correct = 0
        test_error = 0
        noisy_cases = 0
        noisy_cost = 0
        noisy_correct = 0
        noisy_error = 0

        N = len(train_batches) + len(noisy_batches)
        order = range(N)
        np.random.shuffle(order)

        for i in range(N):
            if order[i] < len(train_batches):
                batch = train_batches[order[i]]
                w_noisy = l.weight
                l.weight = w_pure
                epsW_noisy = l.epsW
                l.epsW = 0
                net.train_batch(batch.data, batch.labels, TRAIN)
                l.weight = w_noisy
                l.epsW = epsW_noisy
                cost, correct, num_case = net.get_batch_information()
                train_cases += num_case
                train_correct += correct * num_case
                train_cost += cost * num_case
            else:
                batch = noisy_batches[order[i] - len(train_batches)]
                net.adjust_learning_rate(lrate_beta)
                net.train_batch(batch.data, batch.labels, TRAIN)
                net.adjust_learning_rate(1./lrate_beta)
                if l.epsW > 0:
                    normalize_conf_matrix(net)

                cost, correct, num_case = net.get_batch_information()
                noisy_cases += num_case
                noisy_correct += correct * num_case
                noisy_cost += cost * num_case

        for batch in test_batches:
            w_noisy = l.weight
            l.weight = w_test
            net.train_batch(batch.data, batch.labels, TEST)
            l.weight = w_noisy
            cost, correct, num_case = net.get_batch_information()
            test_cases += num_case
            test_correct += correct * num_case
            test_cost += cost * num_case

        if train_cases > 0:
            train_error = (1. - 1.0*train_correct/train_cases)
            train_cost = (1.0*train_cost/train_cases)
        if noisy_cases > 0:
            noisy_error = (1. - 1.0*noisy_correct/noisy_cases)
            noisy_cost = (1.0*noisy_cost/noisy_cases)
        if test_cases > 0:
            test_error = (1. - 1.0*test_correct/test_cases)
            test_cost = (1.0*test_cost/test_cases)

        print '%d %0.3f %0.3f %0.3f %0.3f %0.3f %0.3f' % (epoch, train_cost, train_error, \
            noisy_cost, noisy_error, test_cost, test_error)

        if net.stat.has_key('test-error') == False:
            net.stat['test-error'] = list()
        net.stat['test-error'].append(test_error)
        #show_stat(net, test_batches)


def train_single(net, num_epoch, train_batches, test_batches):
    l = net.layers[-2]
    M = l.weight.shape[0]
    w_test = np.eye(11)
    w_test = data_loader.copy_to_gpu(w_test)

    if hasattr(net, 'stat') == False:
        net.stat = dict()

    for epoch in range(num_epoch):
        train_cases = 0
        train_cost = 0
        train_correct = 0
        train_error = 0
        test_cases = 0
        test_cost = 0
        test_correct = 0
        test_error = 0

        N = len(train_batches)
        order = range(N)
        np.random.shuffle(order)

        for i in range(N):
			batch = train_batches[order[i]]
			net.train_batch(batch.data, batch.labels, TRAIN)
			cost, correct, num_case = net.get_batch_information()
			train_cases += num_case
			train_correct += correct * num_case
			train_cost += cost * num_case

			if l.epsW > 0:
				normalize_conf_matrix(net)
		
        for batch in test_batches:
            w_noisy = l.weight
            l.weight = w_test
            net.train_batch(batch.data, batch.labels, TEST)
            l.weight = w_noisy
            cost, correct, num_case = net.get_batch_information()
            test_cases += num_case
            test_correct += correct * num_case
            test_cost += cost * num_case

        if train_cases > 0:
            train_error = (1. - 1.0*train_correct/train_cases)
            train_cost = (1.0*train_cost/train_cases)
        if test_cases > 0:
            test_error = (1. - 1.0*test_correct/test_cases)
            test_cost = (1.0*test_cost/test_cases)

        print '%d %0.3f %0.3f %0.3f %0.3f' % (epoch, train_cost, train_error, \
            test_cost, test_error)

        if net.stat.has_key('test-error') == False:
            net.stat['test-error'] = list()
        net.stat['test-error'].append(test_error)
        #show_stat(net, test_batches)
