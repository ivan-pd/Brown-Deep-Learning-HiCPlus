import os,sys
from hicplus import model


## Torch Related Dependencies
# from torch.utils import data
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.autograd import Variable

import tensorflow as tf



## Non-Torch Related Dependencies 
import straw
from scipy.sparse import csr_matrix, coo_matrix, vstack, hstack
from scipy import sparse
import numpy as np
from hicplus import utils
from time import gmtime, strftime
from datetime import datetime
import argparse





startTime = datetime.now()

use_gpu = 0 #opt.cuda
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#if use_gpu and not torch.cuda.is_available():
#    raise Exception("No GPU found, please run without --cuda")

def predict(M,N,inmodel):

    prediction_1 = np.zeros((N, N))

    for low_resolution_samples, index in utils.divide(M):

        #print(index.shape)

        batch_size = low_resolution_samples.shape[0] #256

        # IPD: data.TensorDataset(input, targets) also think of inputs as (x_train, y_train)
        # lowres_set = data.TensorDataset(torch.from_numpy(low_resolution_samples), torch.from_numpy(np.zeros(low_resolution_samples.shape[0])))
        inputs = tf.convert_to_tensor(low_resolution_samples)
        targets = tf.convert_to_tensor(np.zeros(low_resolution_samples.shape[0]))
        lowres_set = tf.data.DataSet.from_tensor_slices((inputs, targets))

        try:
            # DataLoader(dataset, batchsize, shuffle)
            #lowres_loader = torch.utils.data.DataLoader(lowres_set, batch_size=batch_size, shuffle=False)
            lowres_loader = lowres_set.batch(batch_size)
            lowres_loader = lowres_loader.make_one_shot_iterator()
        except:
            continue

        hires_loader = lowres_loader

        m = model.Net(40, 28)
        # m.load_state_dict(torch.load(inmodel, map_location=torch.device('cpu')))
        with tf.device('/cpu:0'):
            model = tf.keras.models.load_model(inmodel)


        # if torch.cuda.is_available():
        #     m = m.cuda()

        # # IPD: Will have to ask about how to run a keras model on a gpu. However,
        # # I think we might not need to have this step if we have a gpu avalible
        # # since there is a setup to run the model on gpu...I think...
        # # is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
        # if is_cuda_gpu_available:
        #     m = m.cuda()

        for i, v1 in enumerate(lowres_loader):
            _lowRes, _ = v1
            # _lowRes = Variable(_lowRes).float()
            _lowRes = tf.Variable(_lowRes)
            if use_gpu:
                _lowRes = _lowRes.cuda()
            y_prediction = m(_lowRes)


        y_predict = y_prediction.data.cpu().numpy()


        # recombine samples
        length = int(y_predict.shape[2])
        y_predict = np.reshape(y_predict, (y_predict.shape[0], length, length))


        for i in range(0, y_predict.shape[0]):

            x = int(index[i][1])
            y = int(index[i][2])
            #print np.count_nonzero(y_predict[i])
            prediction_1[x+6:x+34, y+6:y+34] = y_predict[i]

    return(prediction_1)

def chr_pred(hicfile, chrN1, chrN2, binsize, inmodel):

    # M is just a dense CRS matrix
    M = utils.matrix_extract(chrN1, chrN2, binsize, hicfile) 
    #print(M.shape)
    N = M.shape[0]

    chr_Mat = predict(M, N, inmodel)


#     if Ncol > Nrow:
#         chr_Mat = chr_Mat[:Ncol, :Nrow]
#         chr_Mat = chr_Mat.T
#     if Nrow > Ncol: 
#         chr_Mat = chr_Mat[:Nrow, :Ncol]
#     print(dat.head())       
    return(chr_Mat)



def writeBed(Mat, outname,binsize, chrN1,chrN2):
    with open(outname,'w') as chrom:
        r, c = Mat.nonzero()
        for i in range(r.size):
            contact = int(round(Mat[r[i],c[i]]))
            if contact == 0:
                continue
            #if r[i]*binsize > Len1 or (r[i]+1)*binsize > Len1:
            #    continue
            #if c[i]*binsize > Len2 or (c[i]+1)*binsize > Len2:
            #    continue
            line = [chrN1, r[i]*binsize, (r[i]+1)*binsize,
               chrN2, c[i]*binsize, (c[i]+1)*binsize, contact]
            chrom.write('chr'+str(line[0])+':'+str(line[1])+'-'+str(line[2])+
                     '\t'+'chr'+str(line[3])+':'+str(line[4])+'-'+str(line[5])+'\t'+str(line[6])+'\n')




def main(args):

    '''
    #### Arguments passed into parser (Delete later just for current understanding of variables)
        
        hicplus pred_chromosome
    usage: hicplus pred_chromosome [-h] [-i INPUTFILE] [-m MODEL] [-b BINSIZE] -c
                                   chrN1 chrN2

    optional arguments:
      -h, --help            show this help message and exit
      -i INPUTFILE, --inputfile INPUTFILE
                            path to a .hic file.
      -o OUTPUTFILE, --outputfile OUTPUTFILE
                            path to an output file.
      -m MODEL, --model MODEL
                            path to a model file.
      -b BINSIZE, --binsize BINSIZE
                            predicted resolustion, e.g.10kb, 25kb...,
                            default=10000
      -c chrN1 chrN2, --chrN chrN1 chrN2
                            chromosome number
    '''

    chrN1, chrN2 = args.chrN 
    binsize = args.binsize
    inmodel = args.model
    hicfile = args.inputfile
    #name = os.path.basename(inmodel).split('.')[0]
    #outname = 'chr'+str(chrN1)+'_'+name+'_'+str(binsize//1000)+'pred.txt'
    outname = args.outputfile
    Mat = chr_pred(hicfile,chrN1,chrN2,binsize,inmodel) # Predicted HiC Map
    print(Mat.shape)
    writeBed(Mat, outname, binsize,chrN1, chrN2)
        #print(enhM.shape)

if __name__ == '__main__':
    main()

print(datetime.now() - startTime)
