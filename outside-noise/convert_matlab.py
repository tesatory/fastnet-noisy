import scipy.io
import cPickle
import glob
import sys

path = sys.argv[1]
model_files = glob.glob(path + '/' + 'model_*')
for p in model_files:
	if p[-4:-1] == '.mat':
		continue
	f =  open(p, 'rb')
	net = cPickle.load(f)
	f.close()
	d = dict()
	for ld in net['layers']:
		d['weight_' + str(ld['ind'])] = ld['weight'].copy()
		d['bias_' + str(ld['ind'])] = ld['bias'].copy()
	scipy.io.savemat(p + '.mat', d)
	print 'processed', p