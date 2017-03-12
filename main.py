import find_mxnet
import mxnet as mx
from sklearn.cross_validation import train_test_split
import logging
from kfkd import load2d, augment
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pdb as pdb
from face_models import *
import os


# lenet2 model
# eg..
# python ./main.py -data /home/piyush/Downloads/Kaggle/Facial-Keypoints/ -epochs 30 -model lenet2 -batch 64 -augment -gpus 0,1,2,3

# NO AUGMENT:: 2016-08-08 22:00:32,667 - Epoch[29] Validation-rmse=0.070147
#    AUGMENT:: 2016-08-08 21:59:52,200 - Epoch[29] Validation-rmse=0.065981

def display(img, pts):
    plt.imshow(img , cmap=plt.cm.gray)
    plt.plot(pts[:,0], pts[:,1], '.')
    #pdb.set_trace()
    plt.show()


def parseOptions():
    parser = argparse.ArgumentParser(description='Main face keypoints model runner.')
    parser.add_argument('-model', default='mlp', dest='model', choices = ['mlp', 'lenet2', 'net3'], help='Model to run')
    parser.add_argument('-data', default='none', dest='dataDir', help='Directory where data is stored')
    parser.add_argument('-epochs', type=int, dest='epochs', default=100, help='Number of epochs')
    parser.add_argument('-batch', type=int, dest='batchSize', default=64, help='Batch size')
    parser.add_argument('-gpus', type=str, dest='gpus', default='0', help='List of GPUs to run against i.e. 0,1,2')
    parser.add_argument("-augment", default=False, action="store_true", help="Augment training set")

    opts = parser.parse_args()
    return opts

opts = parseOptions()

#
# logging
#
head = '%(asctime)-15s - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

#
# Load data
#
FTRAIN = os.path.join(opts.dataDir, 'training.csv')
FTEST  = os.path.join(opts.dataDir, 'test.csv')
X, y, = load2d(FTRAIN=FTRAIN, FTEST=FTEST)

if opts.augment:
    print "Augmenting data"
    X, y = augment(X, y) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)
trainIter = mx.io.NDArrayIter(data = X_train, label = y_train, batch_size = opts.batchSize)
valIter   = mx.io.NDArrayIter(data = X_test , label = y_test , batch_size = opts.batchSize)

#
# Select model
#
if opts.model=='mlp':
    net = mlp()
elif opts.model=='lenet2':
    net = lenet2()
elif opts.model=='net3':
    net = net3()

gpus = [mx.gpu(int(gpu)) for gpu in opts.gpus.split(',')]

model   = mx.model.FeedForward(
            ctx             = gpus,
            symbol          = net,
            num_epoch       = opts.epochs,
            momentum        = 0.9,
            learning_rate   = 0.01)

#
# Train
#
model.fit(  X=trainIter, 
            eval_data=valIter, 
            batch_end_callback=mx.callback.Speedometer(1,50), 
            epoch_end_callback=None, 
            eval_metric='rmse')
