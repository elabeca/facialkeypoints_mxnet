import find_mxnet
import mxnet as mx
from sklearn.cross_validation import train_test_split
import logging
from kfkd import load2d
import numpy as np

# logging
head = '%(asctime)-15s - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

X, y = load2d()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)
trainIter = mx.io.NDArrayIter(data = X_train, label = y_train, batch_size = 64)
valIter   = mx.io.NDArrayIter(data = X_test , label = y_test , batch_size = 64)

data 	= mx.symbol.Variable('data')
Y 		= mx.symbol.Variable('softmax_label')

# first convolution
conv1	= mx.symbol.Convolution(data=data, kernel=(3,3), num_filter=32)
act1	= mx.symbol.Activation(data=conv1, act_type='tanh')
pool1	= mx.symbol.Pooling(data=act1, pool_type='max',
							kernel=(2,2), stride=(2,2))

# second convolution
conv2	= mx.symbol.Convolution(data=pool1, kernel=(2,2), num_filter=64)
act2	= mx.symbol.Activation(data=conv2, act_type='tanh')
pool2	= mx.symbol.Pooling(data=act2, pool_type='max',
							kernel=(2,2), stride=(2,2))

# third convolution
conv3	= mx.symbol.Convolution(data=pool2, kernel=(2,2), num_filter=128)
act3	= mx.symbol.Activation(data=conv3, act_type='tanh')
pool3	= mx.symbol.Pooling(data=act3, pool_type='max',
							kernel=(2,2), stride=(2,2))
# first fully connected
flatten = mx.symbol.Flatten(data=pool3)
hidden4	= mx.symbol.FullyConnected(data=flatten, num_hidden=500)
act4	= mx.symbol.Activation(data=hidden4,act_type='tanh')

# second fully connected
hidden5	= mx.symbol.FullyConnected(data=act4, num_hidden=500)
act5	= mx.symbol.Activation(data=hidden5, act_type='tanh')

# output layer
output	= mx.symbol.FullyConnected(data=act5, num_hidden=30)

# loss function
net3	= mx.symbol.LinearRegressionOutput(data=output, label=Y)

model 	= mx.model.FeedForward(
			ctx				= [mx.gpu(0)],
			symbol 			= net3,
			num_epoch 		= 3000,
			momentum		= 0.9,
			learning_rate 	= 0.01)

model.fit(X=trainIter, eval_data=valIter, batch_end_callback=mx.callback.Speedometer(1,50), epoch_end_callback=None, eval_metric='rmse')