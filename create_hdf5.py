#!/usr/bin/python

from PIL import Image
import numpy as np
import sys
from time import time

import h5py

from extract_kanji import *



SIZE_X = 64
SIZE_Y = 63

SAMPLE_WIDTH = 512
TOTAL_RECORDS = 51200 # C1&C2, 320 records
#TOTAL_RECORDS = 50560 # C3, 316 records
#320+320+316=956


DATA_FILE = "etlcdb/ETL8B/ETL8B2C1"
OUTPUT_FILE = "hdf5data/testETL8B2C1_normalized.hdf5"



def progress_bar(counter):
	if counter in range(0, TOTAL_RECORDS, 1000):
		print(counter)
	if (counter+1) == TOTAL_RECORDS:
		print(counter)




ext = Extractor()


t0 = time()
print("Creating hdf5-file...")

# Using int() in order to avoid 'numpy deprecation warnings'
features_training = np.ndarray(shape=(int(TOTAL_RECORDS*0.9), SIZE_Y+1, SIZE_X))
labels_training = np.ndarray(shape=(int(TOTAL_RECORDS*0.9), 1))
features_test = np.ndarray(shape=(int(TOTAL_RECORDS*0.1), SIZE_Y+1, SIZE_X))
labels_test = np.ndarray(shape=(int(TOTAL_RECORDS*0.1), 1))





# Create datasets:
counter_training = 0
counter_test = 0

for counter in range(TOTAL_RECORDS):
	tmp_str = ext.reader(counter+1)
	if (counter%10 == 0):
		features_test[counter_test] = string_to_array(tmp_str)
		labels_test[counter_test] = ext.getJIS(counter+1)
		counter_va += 1
	else:
		features_training[counter_training] = string_to_array(tmp_str)
		labels_training[counter_training] = ext.getJIS(counter+1)
		counter_training += 1

	progress_bar(counter)





features_training = features_training.reshape( (int(TOTAL_RECORDS*0.9), 1, SIZE_Y+1, SIZE_X) )
labels_training = labels_training.reshape( (int(TOTAL_RECORDS*0.9)) )
features_test = features_test.reshape( (int(TOTAL_RECORDS*0.1), 1, SIZE_Y+1, SIZE_X) )
labels_test = labels_test.reshape( (int(TOTAL_RECORDS*0.1)) )



"""
	Define a target variable
	Export features to .hdf5
"""

with h5py.File(OUTPUT_FILE, "w") as file_handle:
	file_handle.create_dataset('training set/features', data=features_training, compression='gzip')
	file_handle.create_dataset('training set/labels', data=labels_training, compression='gzip')
	file_handle.create_dataset('test set/features', data=features_test, compression='gzip')
	file_handle.create_dataset('test set/labels', data=labels_test, compression='gzip')


print("Creation time:" , round(time()-t0, 3), "seconds")
