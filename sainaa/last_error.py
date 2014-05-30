import sys
import zipfile
import cPickle
import numpy as np
import matplotlib.pyplot as plt
import pylab
import pandas

def _load_series(data, scale=1):
    lp = [t['logprob'] for t,count,elapsed in data]
    counts = np.array([count for t,count,elapsed in data]).cumsum()
    examples = counts * scale
    
    elapsed = np.array([elapsed for t,count,elapsed in data])
    logprob = np.array([t[0] for t in lp])
    prec = np.array([t[1] for t in lp])
    return pandas.DataFrame({'lp' : logprob, 'pr' : prec, 'elapsed' : elapsed, 'examples' : examples})

pylab.ion()

chkpnt_path = sys.argv[1]
zf = zipfile.ZipFile(chkpnt_path, 'r')
test_outputs = cPickle.load(zf.open('test_outputs'))
test_df = _load_series(test_outputs)

batch_per_epoch = 2.7 * 1e+6
test_error = pandas.Series(test_df['pr'].values, test_df['examples'].values * 100 / batch_per_epoch)

m = pandas.rolling_mean(test_error, 1000)
print m[-1]

