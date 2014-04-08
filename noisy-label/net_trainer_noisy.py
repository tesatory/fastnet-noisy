import fastnet.net
from fastnet.layer import TRAIN, TEST
from fastnet.cuda_kernel import *
from pycuda import gpuarray
import data_loader
import numpy as np
import matplotlib.pylab as plt
import confusion_matrix
import time

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
    w = data_loader.copy_to_cpu(l.weight)
    if l.weight.shape[0] == 22:
        w = w[11:22,:]

    if hasattr(l, 'entropy_cost'):
        w -= l.epsW * l.entropy_cost * np.log(w + 0.0001)
    if hasattr(l, 'trace_cost'):
        w -= l.epsW * l.trace_cost * (np.eye(w.shape[0]) - 0.1)
    for i in range(w.shape[1]):
        w[:,i] = prob_project(w[:,i])

    if l.weight.shape[0] == 22:
        w = np.concatenate((np.eye(11), w), axis = 0).copy()
    l.weight = data_loader.copy_to_gpu(w)

def update_denoise_matrix(net, labels):
    x = data_loader.copy_to_cpu(net.outputs[-1])
    l = data_loader.copy_to_cpu(labels)
    ww = data_loader.copy_to_cpu(net.W_denoise)
    w = ww[11:21,11:21]
    b = np.zeros((10,10))
    for i in range(l.shape[0]):
        if l[i] > 10:
            b[:,int(l[i]) - 11] += x[11:21,i] 
    for i in range(w.shape[1]):
        if b[:,i].sum() == 0:
            b[:,i] = w[:,i]
    b = b / b.sum(axis = 0, keepdims = True)
    w += net.eps1 * (b - w + net.eps2 * np.eye(w.shape[0]))
    for i in range(w.shape[1]):
        w[:,i] = prob_project(w[:,i])
    ww[11:21,11:21] = w
    net.W_denoise = data_loader.copy_to_gpu(ww)

def train_denoise_matrix(net, num_epoch, train_batches):
    for epoch in range(num_epoch):
        for batch in train_batches:
            net.train_batch(batch.data, batch.labels, TEST)
            update_denoise_matrix(net, batch.labels)


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

    # if hasattr(net, 'W_denoise'):
    #     f.add_subplot('224')
    #     w = data_loader.copy_to_cpu(net.W_denoise)
    #     plt.imshow(w, interpolation = 'nearest')
    #     plt.colorbar()

    plt.draw()
    time.sleep(0.05)


# def train(net, num_epoch, train_batches, test_batches):
#     l = net.layers[-2];
#     w_test = np.zeros(l.weight.shape)
#     w_test[0:10,0:10] = np.eye(10)
#     w_test = data_loader.copy_to_gpu(w_test)

#     if hasattr(net, 'stat') == False:
#         net.stat = dict()

#     for epoch in range(num_epoch):
#         time_sta = time.time()
#         for batch in train_batches:
#             net.train_batch(batch.data, batch.labels, TRAIN)
#             if l.epsW > 0:
#                 normalize_conf_matrix(net)
#         (cost, correct, n) = net.get_batch_information()
#         train_error = 1 - correct
#         train_cost = cost
 
#         for batch in test_batches:
#             w_noisy = l.weight
#             l.weight = w_test
#             net.train_batch(batch.data, batch.labels, TEST)
#             l.weight = w_noisy
#         (cost, correct, n) = net.get_batch_information()
#         test_error = 1 - correct
#         test_cost = cost

#         if net.stat.has_key('test-error') == False:
#             net.stat['test-error'] = list()
#         net.stat['test-error'].append(test_error)
#         print '%d %0.3f %0.3f %0.3f %0.3f' % (epoch, train_cost, train_error, test_cost, test_error),
#         print (time.time() - time_sta)
#         # show_stat(net, test_batches)

def train(net, num_epoch, train_batches, noisy_batches, test_batches, project, lrate_alpha, lrate_beta):
    l = net.layers[-2]
    tmp = gpuarray.empty(l.weight.shape, dtype=np.float32)
    tmp2 = gpuarray.empty((1, l.weight.shape[1]), dtype=np.float32)
    M = l.weight.shape[0]
    w_pure = np.zeros(l.weight.shape)
    w_pure[:10,:10] = np.eye(10)
    if w_pure.shape[0] > 10:
        w_pure[10,10:] = 1
    w_pure = data_loader.copy_to_gpu(w_pure)
    w_test = np.zeros(l.weight.shape)
    w_test[:10,:10] = np.eye(10)
    if w_test.shape[0] > 10:
        w_test[:10,10:] = 0.1
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
                net.adjust_learning_rate(lrate_alpha)
                net.train_batch(batch.data, batch.labels, TRAIN)
                net.adjust_learning_rate(1./lrate_alpha)
                l.weight = w_noisy
                l.epsW = epsW_noisy
                cost, correct, num_case = net.get_batch_information()
                train_cases += num_case
                train_correct += correct * num_case
                train_cost += cost * num_case
            else:
                batch = noisy_batches[order[i] - len(train_batches)]
                if lrate_beta > 0:
                    net.adjust_learning_rate(lrate_beta)
                    net.train_batch(batch.data, batch.labels, TRAIN)
                    net.adjust_learning_rate(1./lrate_beta)
                else:
                    net.train_batch(batch.data, batch.labels, TEST)
                if l.epsW > 0:
                    # normilize confussion layer
                    if project == False:
                        relu_activate(l.weight, tmp, 0)
                        tmp2.fill(0)
                        add_col_sum_to_vec(tmp2, tmp)
                        div_vec_to_cols(tmp, tmp2, l.weight)
                    else:
                        normalize_conf_matrix(net)

                        # tmp2.fill(0)
                        # add_col_sum_to_vec(tmp2, l.weight)
                        # add_vec_to_cols(l.weight, tmp2 - 1, alpha = -1./M, beta = 1)
                        # relu_activate(l.weight, tmp, 0)
                        # tmp2.fill(0)
                        # add_col_sum_to_vec(tmp2, tmp)
                        # div_vec_to_cols(tmp, tmp2, l.weight)

                cost, correct, num_case = net.get_batch_information()
                noisy_cases += num_case
                noisy_correct += correct * num_case
                noisy_cost += cost * num_case

            # w = data_loader.copy_to_cpu(net.layers[-4].weight)
            # w[10,:] = 0
            # net.layers[-4].weight = data_loader.copy_to_gpu(w)
            # b = data_loader.copy_to_cpu(net.layers[-4].bias)
            # b[10] = 1                        
            # net.layers[-4].bias = data_loader.copy_to_gpu(b)

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
        show_stat(net, train_batches)

