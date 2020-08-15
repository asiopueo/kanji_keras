from sklearn.model_selection import train_test_split
import numpy as np
import h5py
from keras.utils import np_utils, to_categorical
from keras.callbacks import CSVLogger
from nets import *
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


HDF5_DATA = 'hdf5data/ETL8B.hdf5'
MODEL = 'models/model_M6_2_total.hdf5'
LOG_FILE = './logs/M6_2_total.csv'


with h5py.File(HDF5_DATA, 'r') as f:
	X_train = f['training_set/features'].value
	y_train = to_categorical(f['training_set/labels'].value)

	X_test = f['test_set/features'].value
	y_test = to_categorical(f['test_set/labels'].value)

	X_train, _, y_train, _= train_test_split(X_train, y_train, test_size=0.0, random_state=1)


	csv_logger = CSVLogger(LOG_FILE)


	model = model_M6_2()
	#model.load_weights('models/model_M6_2.hdf5')
	model.fit(X_train, y_train, batch_size=128, epochs=40, shuffle=True, verbose=1, validation_split=0.2, callbacks=[csv_logger])

	loss, acc = model.evaluate(X_test, y_test, verbose=0)
	#print(model.metrics_names)
	print('Loss on test set: ', loss)
	print('Accuracy on test set: ', acc)

	model.save_weights(MODEL)



