import fastnet.net
from fastnet.layer import TRAIN, TEST
from fastnet.cuda_kernel import *
from pycuda import gpuarray

def train(net, num_epoch, train_batches, test_batches):
    for epoch in range(num_epoch):
        time_sta = time.time()
        for batch in train_batches:
            net.train_batch(batch.data, batch.labels, TRAIN)
        (cost, correct, n) = net.get_batch_information()
        train_error = 1 - correct
        train_cost = cost
 
        for batch in test_batches:
            net.train_batch(batch.data, batch.labels, TEST)
        (cost, correct, n) = net.get_batch_information()
        test_error = 1 - correct
        test_cost = cost

        print '%d %0.3f %0.3f %0.3f %0.3f' % (epoch, train_cost, train_error, test_cost, test_error),
        print (time.time() - time_sta)
        # show_stat(net, test_batches)