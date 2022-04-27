# Author: Yan Zhang
# Email: zhangyan.cse (@) gmail.com

import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import gzip
import model
from torch.utils import data
import torch
import torch.optim as optim
from torch.autograd import Variable
from time import gmtime, strftime
import sys
import torch.nn as nn


import tensorflow as tf


use_gpu = 1

conv2d1_filters_numbers = 8
conv2d1_filters_size = 9
conv2d2_filters_numbers = 8
conv2d2_filters_size = 1
conv2d3_filters_numbers = 1
conv2d3_filters_size = 5


down_sample_ratio = 16
epochs = 10
HiC_max_value = 100



# This block is the actual training data used in the training. The training data is too large to put on Github, so only toy data is used.
# cell = "GM12878_replicate"
# chrN_range1 = '1_8'
# chrN_range = '1_8'

# low_resolution_samples = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/'+cell+'down16_chr'+chrN_range+'.npy.gz', "r")).astype(np.float32) * down_sample_ratio
# high_resolution_samples = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/original10k/'+cell+'_original_chr'+chrN_range+'.npy.gz', "r")).astype(np.float32)

# low_resolution_samples = np.minimum(HiC_max_value, low_resolution_samples)
# high_resolution_samples = np.minimum(HiC_max_value, high_resolution_samples)


low_resolution_samples = np.load(gzip.GzipFile('../../data/GM12878_replicate_down16_chr19_22.npy.gz', "r")).astype(np.float32) * down_sample_ratio

low_resolution_samples = np.minimum(HiC_max_value, low_resolution_samples)

batch_size = low_resolution_samples.shape[0]

# Reshape the high-quality Hi-C sample as the target value of the training.
sample_size = low_resolution_samples.shape[-1]
padding = conv2d1_filters_size + conv2d2_filters_size + conv2d3_filters_size - 3
half_padding = padding / 2
output_length = sample_size - padding


print(low_resolution_samples.shape)

# lowres_set = data.TensorDataset(torch.from_numpy(low_resolution_samples), torch.from_numpy(np.zeros(low_resolution_samples.shape[0])))
# lowres_loader = torch.utils.data.DataLoader(lowres_set, batch_size=batch_size, shuffle=False)
inputs = tf.convert_to_tensor(low_resolution_samples)
targets = tf.convert_to_tensor(np.zeros(low_resolution_samples.shape[0]))
lowres_set = tf.data.DataSet.from_tensor_slices((inputs, targets))
lowres_loader = lowres_set.batch(batch_size)
lowres_loader = lowres_loader.make_one_shot_iterator()


production = False
try:
    high_resolution_samples = np.load(gzip.GzipFile('../../data/GM12878_replicate_original_chr19_22.npy.gz', "r")).astype(np.float32)
    high_resolution_samples = np.minimum(HiC_max_value, high_resolution_samples)
    Y = []
    for i in range(high_resolution_samples.shape[0]):
        no_padding_sample = high_resolution_samples[i][0][half_padding:(sample_size-half_padding) , half_padding:(sample_size - half_padding)]
        Y.append(no_padding_sample)
    Y = np.array(Y).astype(np.float32)

    # hires_set = data.TensorDataset(torch.from_numpy(Y), torch.from_numpy(np.zeros(Y.shape[0])))
    # hires_loader = torch.utils.data.DataLoader(hires_set, batch_size=batch_size, shuffle=False)

    hires_inputs = tf.convert_to_tensor(Y)
    hires_targets = tf.convert_to_tensor(np.zeros(Y.shape[0]))
    hires_set = tf.data.DataSet.from_tensor_slices((hires_inputs, hires_targets))
    hires_loader = hires_set.batch(batch_size)
    hires_loader = hires_loader.make_one_shot_iterator()

except:
    production = True
    hires_loader = lowres_loader

Net = model.Net(40, 28)
# Net.load_state_dict(torch.load('../model/pytorch_model_12000'))

# Have to instantiate model
# Net = model.Net()
# tf.keras.models.load_model('../model/pytorch_model_12000')
Net.load_weights('../model/pytorch_model_12000')

if use_gpu:
    Net = Net.cuda() #IPD: Not sure about the GPU stuff again

# _loss = nn.MSELoss()
_loss = tf.keras.losses.MeanSquaredError()

running_loss = 0.0
running_loss_validate = 0.0
reg_loss = 0.0


for i, (v1, v2) in enumerate(zip(lowres_loader, hires_loader)):
    _lowRes, _ = v1
    _highRes, _ = v2


    _lowRes = tf.Variable(_lowRes)
    _highRes = tf.Variable(_highRes)


    if use_gpu:
        _lowRes = _lowRes.cuda()
        _highRes = _highRes.cuda()

    ## I think this might be the gpu equivalent:
    # is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
    # if is_cuda_gpu_available and use_gpu:
    #    _lowRes = _lowRes.cuda()
    #    _highRes = _highRes.cuda()


    # y_prediction = Net(_lowRes)
    # if (not production):
    #     loss = _loss(_highRes, y_prediction)
    y_prediction = model.call(_lowRes)
    if (not production):
        loss = _loss(y_prediction, _highRes)


    # running_loss += loss.data[0]
    running_loss += loss

print('-------', i, running_loss, strftime("%Y-%m-%d %H:%M:%S", gmtime()))

# y_prediction = y_prediction.data.cpu().numpy()

print("Is there a GPU available: "),
print(tf.config.list_physical_devices("GPU"))

print("Is the Tensor on GPU #0:  "),
print(y_prediction.device.endswith('GPU:0'))

with tf.device("CPU:0"):
  assert y_prediction.device.endswith("CPU:0")
  y_prediction.numpy()


print(y_prediction.shape)
