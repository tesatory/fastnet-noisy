#!/usr/bin/python2.7
'''
This test is for naive trainer to traine a full imagenet model
'''

from fastnet import data, trainer, net, parser

test_id = 'noisy-eye'

data_dir = '/ssd/fergusgroup/sainaa/imagenet/train/'
data_dir_noisy = '/ssd/fergusgroup/sainaa/imagenet/noisy/'
checkpoint_dir = '/ssd/fergusgroup/sainaa/imagenet/checkpoint/'
param_file = '/home/ss7345/fastnet-noisy/config/imagenet-noisy.cfg'
output_dir = ''
output_method = 'disk'

train_range = range(101, 1301) #1,2,3,....,40
train_range_noisy = range(1, 2000) #1,2,3,....,40
test_range = range(1, 101) #41, 42, ..., 48
data_provider = 'imagenet'

train_dp_clear = data.get_by_name(data_provider)(data_dir,train_range)
train_dp_noisy = data.get_by_name(data_provider)(data_dir_noisy,train_range_noisy)
train_dp = data.NoisyDataProvider(train_dp_clear, train_dp_noisy)
test_dp = data.get_by_name(data_provider)(data_dir, test_range)
checkpoint_dumper = trainer.CheckpointDumper(checkpoint_dir, test_id)

init_model = checkpoint_dumper.get_checkpoint()
if init_model is None:
  init_model = parser.parse_config_file(param_file)

save_freq = 100
test_freq = 100
adjust_freq = 100
factor = 1
num_epoch = 10
learning_rate = 0.1
batch_size = 128
image_color = 3
image_size = 224
image_shape = (image_color, image_size, image_size, batch_size)

net = net.FastNet(learning_rate, image_shape, init_model)

param_dict = globals()
t = trainer.Trainer(**param_dict)
net.adjust_learning_rate(0.1)
t.train(num_epoch)

