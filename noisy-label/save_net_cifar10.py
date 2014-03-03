import scipy.io
import data_loader

def save_net(net, path):
	d = dict()
	d['w1'] = data_loader.copy_to_cpu(net.layers[1].weight)
	d['w2'] = data_loader.copy_to_cpu(net.layers[5].weight)
	d['w3'] = data_loader.copy_to_cpu(net.layers[9].weight)
	d['w4'] = data_loader.copy_to_cpu(net.layers[12].weight)
	d['b1'] = data_loader.copy_to_cpu(net.layers[1].bias)
	d['b2'] = data_loader.copy_to_cpu(net.layers[5].bias)
	d['b3'] = data_loader.copy_to_cpu(net.layers[9].bias)
	d['b4'] = data_loader.copy_to_cpu(net.layers[12].bias)
	if len(net.layers) == 16:
		d['confw'] = data_loader.copy_to_cpu(net.layers[14].weight)
	if hasattr(net, 'W_denoise'):
		d['W_denoise'] = data_loader.copy_to_cpu(net.W_denoise)
	scipy.io.savemat(path, d)

def load_net(net, path):
	d = scipy.io.loadmat(path)
	net.layers[1].weight = data_loader.copy_to_gpu(d['w1'].copy())
	net.layers[5].weight = data_loader.copy_to_gpu(d['w2'].copy())
	net.layers[9].weight = data_loader.copy_to_gpu(d['w3'].copy())
	net.layers[12].weight = data_loader.copy_to_gpu(d['w4'].copy())
	net.layers[1].bias = data_loader.copy_to_gpu(d['b1'].copy())
	net.layers[5].bias = data_loader.copy_to_gpu(d['b2'].copy())
	net.layers[9].bias = data_loader.copy_to_gpu(d['b3'].copy())
	net.layers[12].bias = data_loader.copy_to_gpu(d['b4'].copy())
	if 'conf-w' in d and len(net.layers) == 16:
		net.layers[14].weight = data_loader.copy_to_gpu(d['conf-w'].copy())
	if 'confw' in d and len(net.layers) == 16:
		net.layers[14].weight = data_loader.copy_to_gpu(d['confw'].copy())
	if 'W_denoise' in d:
		net.W_denoise = data_loader.copy_to_gpu(d['W_denoise'].copy())

