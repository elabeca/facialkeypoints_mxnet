import find_mxnet
import mxnet as mx
from sklearn.cross_validation import train_test_split
import logging
from kfkd import load
import numpy as np

# logging
head = '%(asctime)-15s - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

X, y = load()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)
trainIter = mx.io.NDArrayIter(data = X_train, label = y_train, batch_size = 64)
valIter   = mx.io.NDArrayIter(data = X_test , label = y_test , batch_size = 64)

data 	= mx.symbol.Variable('data')
flatten = mx.symbol.Flatten(data=data)
Y 		= mx.symbol.Variable('softmax_label')

fc1		= mx.symbol.FullyConnected(data=flatten, num_hidden=100)
act1	= mx.symbol.Activation(data=fc1, act_type='relu')
fc2		= mx.symbol.FullyConnected(data=act1, num_hidden=30)
mlp		= mx.symbol.LinearRegressionOutput(data=fc2, label=Y)

model 	= mx.model.FeedForward(
			ctx				= [mx.gpu(0)],
			symbol 			= mlp,
			num_epoch 		= 400,
			momentum		= 0.9,
			learning_rate 	= 0.01)

model.fit(X=trainIter, eval_data=valIter, batch_end_callback=mx.callback.Speedometer(1,50), epoch_end_callback=None, eval_metric='rmse')