#################################################################
#
#	Definition File for all CNNs used for the Kanji Recognizer
#
#################################################################


import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import MaxPooling2D, Flatten, BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.utils import np_utils, to_categorical
from keras.optimizers import Adam



# Included solely for illustrating purposes:
def model_MNIST():
	model = Sequential()
	model.add(Dense(512, input_shape=(784,)))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(10))
	model.add(Activation('softmax'))
	return model



# Model M6-1
def model_M6_1():
	model = Sequential()	

	# input_shape=(1,64,64)
	model.add( Convolution2D(32, (3,3), input_shape=(1,64,64), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') ) # conv3-32
	# shape = ()
	model.add( Convolution2D(32, (3,3), padding='valid', activation='relu', data_format='channels_first') ) # conv3-32
	# shape = ()
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )
	# shape = ()
	model.add( Convolution2D(64, (3,3), padding='valid', activation='relu', data_format='channels_first') ) # conv3-64
	# shape = ()
	model.add( Convolution2D(64, (3,3), padding='valid', activation='relu', data_format='channels_first') ) # conv3-64
	# shape = ()
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )
	# shape = ()
	model.add( Flatten() )
	# shape = ()
	model.add( Dense(256) )
	# shape = ()
	model.add( Dense(320) )
	# shape = ()
	model.add( Activation('softmax') )
	
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	return model




# Model M6-2
def model_M6_2():
	model = Sequential()	

	# input_shape=(1,64,64)
	model.add( Convolution2D(64, (3,3), input_shape=(1,64,64), padding='valid', strides=(1,1), kernel_initializer='normal', activation='relu', data_format='channels_first') ) # conv3-64
	model.add( BatchNormalization() )
	#model.add( Activation('relu') ) 
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )
	model.add( Dropout(0.5) )

	# shape = ()
	model.add( Convolution2D(128, (3,3), padding='valid', kernel_initializer='normal', activation='relu', data_format='channels_first') ) # conv3-128
	model.add( BatchNormalization() )
	#model.add( Activation('relu') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )
	model.add( Dropout(0.5) )
	
	# shape = ()
	model.add( Convolution2D(512, (3,3), padding='valid', kernel_initializer='normal', activation='relu', data_format='channels_first') ) # conv3-512
	model.add( BatchNormalization() )
	#model.add( Activation('relu') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )
	model.add( Dropout(0.5) )

	model.add( Flatten() )
	model.add( Dense(4096, kernel_initializer='random_uniform') )
	model.add( Dropout(0.5) )
	model.add( BatchNormalization() )
	model.add( Dense(4096, kernel_initializer='random_uniform') )
	model.add( Dropout(0.5) )
	model.add( BatchNormalization() )
	model.add( Dense(956, kernel_initializer='random_uniform') )
	model.add( BatchNormalization() )
	model.add( Activation('softmax') )
	
	adam = Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.1, amsgrad=False)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	return model


# Model M7-1
def model_M7_1():
	model = Sequential()
	model.add( Convolution2D(64, (3,3), input_shape=(1,64,64), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(128, (3,3), padding='valid', activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first'))

	model.add( Convolution2D(512, (3,3), padding='valid', activation='relu', data_format='channels_first') )
	model.add( Convolution2D(512, (3,3), padding='valid', activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first'))

	model.add( Flatten() )
	model.add( Dense(4096) )
	model.add( Dense(4096) )
	model.add( Dense(320) )
	model.add( Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam')
	return model


# Model M7-2
def model_M7_2():
	model = Sequential()
	model.add( Convolution2D(64, (3,3), input_shape=(1,64,64), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(128, (3,3), padding='valid', activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(192, (3,3), padding='valid', activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(256, (3,3), padding='valid', activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Flatten() )
	model.add( Dense(1024) )
	model.add( Dense(1024) )
	model.add( Dense(320) )
	model.add( Activation('softmax'))

	return model


# Model M8
def model_M8():
	model = Sequential()
	model.add( Convolution2D(32, (3,3), input_shape=(1,64,64), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Dropout(0.5) )
	model.add( BatchNormalization() )
	model.add( Convolution2D(32, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Dropout(0.5) )
	model.add( BatchNormalization() )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(64, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Dropout(0.5) )
	model.add( BatchNormalization() )
	model.add( Convolution2D(64, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Dropout(0.5) )
	model.add( BatchNormalization() )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(128, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Dropout(0.5) )
	model.add( BatchNormalization() )
	model.add( Convolution2D(128, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Dropout(0.5) )
	model.add( BatchNormalization() )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Flatten() )
	model.add( Dense(1024) )
	model.add( Dropout(0.5) )
	model.add( BatchNormalization() )
	model.add( Dense(320) )
	model.add( BatchNormalization() )
	model.add( Activation('softmax') )

	adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-4, decay=0.1, amsgrad=False)
	model.compile(loss='categorical_crossentropy', optimizer=adam)
	return model


# Model M9
def model_M9():
	model = Sequential()
	model.add( Convolution2D(64, (3,3), input_shape=(1,64,64), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(128, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(256, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(256, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(512, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(512, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Flatten() )
	model.add( Dense(4096) )
	model.add( Dense(4096) )
	model.add( Dense(320) )
	model.add( Activation('softmax'))

	return model


# Model M11
def model_M11():
	model = Sequential()
	model.add( Convolution2D(64, (3,3), input_shape=(1,64,64), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(64, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(128, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(128, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(256, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(256, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(512, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(512, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Flatten() )
	model.add( Dense(1024) )
	model.add( Dense(1024) )
	model.add( Dense(320) )
	model.add( Activation('softmax'))

	return model


# Model M12
def model_M12():
	model = Sequential()
	model.add( Convolution2D(64, (3,3), input_shape=(1,64,64), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(64, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(128, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(128, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(256, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(256, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(512, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(512, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(512, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Flatten() )
	model.add( Dense(4096) )
	model.add( Dense(4096) )
	model.add( Dense(320) )
	model.add( Activation('softmax'))

	return model


# Model M13
def model_M13():
	model = Sequential()
	model.add( Convolution2D(32, (3,3), input_shape=(1,64,64), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(32, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(64, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(64, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(128, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(128, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(256, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(256, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(512, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(512, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Flatten() )
	model.add( Dense(1024) )
	model.add( Dense(1024) )
	model.add( Dense(320) )
	model.add( Activation('softmax'))

	return model


# Model M16
def model_M16():
	model = Sequential()
	model.add( Convolution2D(64, (3,3), input_shape=(1,64,64), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(64, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(128, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(128, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(256, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(256, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(256, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(512, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(512, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(512, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Convolution2D(512, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(512, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( Convolution2D(512, (3,3), padding='valid', strides=(1,1), activation='relu', data_format='channels_first') )
	model.add( MaxPooling2D(pool_size=(2,2), data_format='channels_first') )

	model.add( Flatten() )
	model.add( Dense(4096) )
	model.add( Dense(4096) )
	model.add( Dense(320) )
	model.add( Activation('softmax'))

	return model



