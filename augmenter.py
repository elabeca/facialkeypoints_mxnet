import find_mxnet
import mxnet as mx
from sklearn.cross_validation import train_test_split
import logging
from kfkd import load2d
import numpy as np
import matplotlib.pyplot as plt
import pdb as pdb


def display(img, pts):
    plt.imshow(img , cmap=plt.cm.gray)
    plt.plot(pts[:,0], pts[:,1], '.')
    #pdb.set_trace()
    plt.show()


X, y = load2d()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 42)


I = 1000

# Flip all images LR-wise
Xf = X[:,:,:, ::-1]

# Flip label x-cordinates LR wise
xMid = 96/2
y = y.reshape((np.shape(y)[0], 15, 2))

# Select x values
xVals = y[:,:,0] - xMid
n = xVals <0
p = xVals>=0

# Flip LR wise by changing sign
xVals[n] = np.abs(xVals[n])
xVals[p] = -xVals[p]

# Save to new struct
yf = np.copy(y)
yf[:,:,0] = xVals + xMid

# Have a look
display(X[I,0], y[I])
display(Xf[I,0], yf[I])


