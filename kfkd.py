import os
import numpy as np
import matplotlib.pyplot as pyplot
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import pdb as pdb

def augment(X, y):
  """
  Augment the training set
  """

  flip_indices = [
      (0, 2), (1, 3),
      (4, 8), (5, 9), (6, 10), (7, 11),
      (12, 16), (13, 17), (14, 18), (15, 19),
      (22, 24), (23, 25),
      ]

  def flipData(X, y):

    # Flip all images LR-wise
    Xf = X[:,:,:, ::-1]
  
    # Flip label x-cordinates LR wise
    yf = y.reshape((np.shape(y)[0], 15, 2))
  
    # Flip x coords LR wise by changing sign
    yf[:,:,0] = yf[:,:,0] * -1
  
    # Save to new struct
    yf = yf.reshape((np.shape(yf)[0], 30))
  
    # Swap places, e.g. left_eye_center_x -> right_eye_center_x
    for a, b in flip_indices:
        yf[:, a] =  yf[:, b]
        yf[:, b] =  yf[:, a]

    return Xf, yf

  Xf, yf = flipData(X, y)
  #Xf, yf = flipData(Xf, yf) # get back original
  
  X = np.concatenate((X, Xf), axis=0)
  y = np.concatenate((y, yf), axis=0)

  return X, y

def load(test=False, cols=None, FTEST='', FTRAIN=''):
  """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
  Pass a list of *cols* if you're only interested in a subset of the
  target columns.
  """

  fname = FTEST if test else FTRAIN
  df = read_csv(os.path.expanduser(fname)) # load pandas dataframe

  # The Image column has pixel values seperated by space; convert
  # the values to numpy arrays:
  df['Image'] = df['Image'].apply(lambda img: np.fromstring(img, sep=' '))

  if cols:	# get a subset of columns
    df = df[list(cols) + ['Image']]

  print(df.count())	# prints the number of values for each column
  df = df.dropna()	# drop all rows that have missing values in them

  X = np.vstack(df['Image'].values)
  X = X / 255. 	# scale pixel values to [0, 1]
  X = X.astype(np.float32)

  if not test:	# only FTRAIN has any target columns
    y = df[df.columns[:-1]].values
    y = (y - 48) / 48   # scale target coordinates to [-1, 1]
    X, y = shuffle(X, y, random_state=42)   # shuffle train data
    y = y.astype(np.float32)

    print("X.shape == {}; X.min == {:.3f}, X.max == {:.3f}".format(X.shape, X.min(), X.max()))
    print("y.shape == {}; y.min == {:.3f}, y.max == {:.3f}".format(y.shape, y.min(), y.max()))
  else:
    y = None

  #pdb.set_trace()
  return X, y

def load2d(test=False, cols=None, FTEST='', FTRAIN=''):
  X, y= load(test=test, cols=cols, FTEST=FTEST, FTRAIN=FTRAIN)
  X = X.reshape(-1, 1, 96, 96)
  return X, y

def plot_loss(net):
  train_loss = np.array([i["train_loss"] for i in net.train_history_])
  valid_loss = np.array([i["valid_loss"] for i in net.train_history_])

  pyplot.plot(train_loss, linewidth=3, label="train")
  pyplot.plot(valid_loss, linewidth=3, label="valid")

  pyplot.grid()
  pyplot.legend()
  pyplot.xlabel("epoch")
  pyplot.ylabel("loss")
  pyplot.ylim(1e-3, 1e-2)
  pyplot.yscale("log")
  pyplot.show()

def plot_sample(x, y, axis):
  img = x.reshape(96, 96)
  axis.imshow(img, cmap='gray')
  axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

def float32(k):
  return np.cast['float32'](k)