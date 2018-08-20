import numpy as np

import h5py
from keras.utils import np_utils, to_categorical

from nets import *





with h5py.File('data/testC1_normalized.hdf5', 'r') as f:
	X_train = f['training set/features'].value
	Y_train = to_categorical(f['training set/labels'].value)
	X_test = f['validation set/features'].value
	Y_test = to_categorical(f['validation set/labels'].value)

	model = model_M6_1()
	model.fit(X_train, Y_train, batch_size=512, epochs=12, verbose=1, validation_split=0.1)

	score = model.evaluate(X_test, Y_test, verbose=0)

	print('Test score: ', score)

	model.save_weights('models/model_M6_1.hdf5')



