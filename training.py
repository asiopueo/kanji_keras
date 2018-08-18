import numpy as np

import h5py
from keras.utils import np_utils, to_categorical

from nets import *





with h5py.File('data/testC1_normalized.hdf5', 'r') as f:
	X_train = f['training set/features'].value
	Y_train = to_categorical(f['training set/labels'].value)
	X_validation = f['validation set/features'].value
	Y_validation = to_categorical(f['validation set/labels'].value)

	model = model_M6_1()
	model.fit(X_train, Y_train, batch_size=1024, epochs=12, verbose=1, validation_data=(X_validation, Y_validation), shuffle=True)

	score = model.evaluate(X_validation, Y_validation, verbose=0)

	print('Test score: ', score)

	model.save_weights('models/model_M6_2.hdf5')



