from sklearn.model_selection import train_test_split

import numpy as np
import h5py


print('Hello!')


with h5py.File('hdf5data/ETL8B2C1.hdf5', 'r') as f:
	#X_train = f['training_set/features'].value
	#y_train = to_categorical(f['training_set/labels'].value)

	X_test = f['test_set/features'].value
	y_test = f['test_set/labels'].value


	X_train, _, y_train, _= train_test_split(X_test, y_test, test_size=0.0, random_state=None)

	print(X_train.shape)
	print(X_test.shape)
	print(y_train.shape)
	print(y_test.shape)


