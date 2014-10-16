import scipy.io
import cPickle
import data_loader
import glob

def save_mat(net, path):
	d = dict()
	for l in net.layers:
		if hasattr(l,'weight'):
			d[l.name + '_weight'] = data_loader.copy_to_cpu(l.weight)
			d[l.name + '_bias'] = data_loader.copy_to_cpu(l.bias)
	scipy.io.savemat(path, d)

def save(net, path):
	d = dict()
	d['layers'] = list()
	for i in range(len(net.layers)):
		l = net.layers[i]
		if hasattr(l,'weight'):
			ld = dict()
			ld['weight'] = data_loader.copy_to_cpu(l.weight)
			ld['bias'] = data_loader.copy_to_cpu(l.bias)
			ld['ind'] = i
			d['layers'].append(ld)

	f = open(path, 'wb')
	d['stat'] = net.stat
	cPickle.dump(d, f, protocol=-1)
	f.close()

def load(path, net):
	f =  open(path, 'rb')
	d = cPickle.load(f)
	f.close()
	for ld in d['layers']:
		l = net.layers[ld['ind']]
		l.weight = data_loader.copy_to_gpu(ld['weight'].copy())
		l.bias = data_loader.copy_to_gpu(ld['bias'].copy())
	net.stat = d['stat']

def try_load(net):
	model_files = glob.glob(net.output_dir + 'model_*')
	if len(model_files) == 0:
		return False
	max_epoch = -1
	path = ''
	for f in model_files:
		epoch = int(f.split('_')[-1])
		if epoch > max_epoch:
			path = f
	print 'Loading model from ' + path
	load(path, net)
	return True
