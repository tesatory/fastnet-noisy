import numpy as np
import data_loader

def prob_project(x):
    x = x.copy()
    pos = np.array([True] * x.shape[0])
    eps = 0.0001
    while x.sum() - 1 > eps:
        r = x.sum() - 1
        x[pos] -= r / pos.sum()
        pos = x > 0
        x[~pos] = 0
    return x

def train(w, x, y, lrate, wc, max_epoch):
	for epoch in range(max_epoch):
		z = np.dot(w, x)
		cost = -(np.log((y * z).sum(axis = 0))).mean() + wc * (w**2).sum()
		# print cost
		zg = (-1.0/(z + 0.001)) * y
		wg = np.dot(zg, x.transpose()) / x.shape[1]
		wg += np.dot(wc, w)
		w = w - lrate * wg
		for i in range(w.shape[1]):
			w[:,i] = prob_project(w[:,i])
	return w

def get_labels(net, noisy_batches):
	w = data_loader.copy_to_cpu(net.layers[-2].weight)
	N = len(noisy_batches) * 128
	y = np.zeros([w.shape[0], N], dtype = np.float32)
	c = 0
	for b in noisy_batches:
		labels = data_loader.copy_to_cpu(b.labels)
		for i in range(len(labels)):
			y[int(labels[i]), i + c] = 1
		c += 128
	return y

def get_net_output(net, noisy_batches):
	w = data_loader.copy_to_cpu(net.layers[-2].weight)
	N = len(noisy_batches) * 128
	x = np.zeros([w.shape[1], N], dtype = np.float32)
	c = 0
	for b in noisy_batches:
		data, label = net.prepare_for_train(b.data, b.labels)
		net.fprop(data, net.output, False)
		net_out = data_loader.copy_to_cpu(net.outputs[-3])
		x[:,c:c+128] = net_out
		c += 128
	return x

def update_confusion_matrix(net, noisy_batches, y, wc):
	w = data_loader.copy_to_cpu(net.layers[-2].weight)
	x = get_net_output(net, noisy_batches)
	w = train(w, x, y, 1, wc, 50)
	w = train(w, x, y, 0.1, wc, 10)
	net.layers[-2].weight = data_loader.copy_to_gpu(w)

def get_confusion(net, test_batches):
	x = get_net_output(net, test_batches)
	y = get_labels(net, test_batches)
	w = np.dot(x, y.transpose())
	w = w / w.sum(axis = 0, keepdims = True)
	return w

def	mix_labels(W, labels):
	N = W.shape[0]
	new_labels = np.zeros(labels.size)
	for i in xrange(labels.size):
		r = np.random.multinomial(1, W[:,labels[i]])
		new_labels[i] = (r * range(N)).sum()
	return new_labels
	