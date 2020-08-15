#!/usr/bin/python

from PIL import Image
import numpy as np
import sys
from time import time

import h5py

from extract_kanji import *
from database.ETL8B import Database


SIZE_X = 64
SIZE_Y = 63

SAMPLE_WIDTH = 512
TOTAL_RECORDS_C1_C2 = 51200 # C1&C2, 320 records each
TOTAL_RECORDS_C3 = 50560 # C3, 316 records
TOTAL_RECORDS = 152960 # C1,C2, and C3
#Total number of characters: 320+320+316=956


#DATA_FILE = "etlcdb/ETL8B/ETL8B2C3"
#OUTPUT_FILE = "hdf5data/ETL8B2C3.hdf5"
#DB_File = "database/ETL8B2C3.db"


data_list = [ 'etlcdb/ETL8B/ETL8B2C1', 
			  'etlcdb/ETL8B/ETL8B2C2', 
			  'etlcdb/ETL8B/ETL8B2C3']

records_list = [ 51200,
				 51200,
				 50560]

db_file = 'database/ETL8B.db'

character_set_list = zip(data_list, records_list)

output_file = 'hdf5data/ETL8B.hdf5'



def progress_bar(counter):
	if counter in range(0, TOTAL_RECORDS, 1000):
		print(counter)
	if (counter+1) == TOTAL_RECORDS:
		print(counter)







t0 = time()
print("Creating hdf5-file...")

# Using int() in order to avoid 'numpy deprecation warnings'
features_training = np.ndarray(shape=(int(TOTAL_RECORDS*0.9), SIZE_Y+1, SIZE_X))
labels_training = np.ndarray(shape=(int(TOTAL_RECORDS*0.9),))
features_test = np.ndarray(shape=(int(TOTAL_RECORDS*0.1), SIZE_Y+1, SIZE_X))
labels_test = np.ndarray(shape=(int(TOTAL_RECORDS*0.1),))



db = Database(db_file)

# Create datasets:
counter_training = 0
counter_test = 0

for etl_file, number_of_records in character_set_list:
	ext = Extractor(etl_file)

	for counter in range(number_of_records):
		jis_code = ext.getJIS(counter+1)
		index = db.getIndex( jis_code )
		if (counter%10 == 0):
			features_test[counter_test] = ext.getArray(counter+1)/255.
			labels_test[counter_test] = index # Saves JIS code as integer instead of hex
			counter_test += 1
		else:
			features_training[counter_training] = ext.getArray(counter+1)/255.
			labels_training[counter_training] = index # Saves JIS code as integer instead of hex
			counter_training += 1

		progress_bar(counter_training+counter_test)





features_training = features_training.reshape( (int(TOTAL_RECORDS*0.9), 1, SIZE_Y+1, SIZE_X) )
labels_training = labels_training.reshape( (int(TOTAL_RECORDS*0.9),) )
features_test = features_test.reshape( (int(TOTAL_RECORDS*0.1), 1, SIZE_Y+1, SIZE_X) )
labels_test = labels_test.reshape( (int(TOTAL_RECORDS*0.1),) )



"""
	Define a target variable
	Export features to .hdf5
"""

with h5py.File(output_file, "w") as file_handle:
	file_handle.create_dataset('training_set/features', data=features_training, compression='gzip')
	file_handle.create_dataset('training_set/labels', data=labels_training, compression='gzip')
	file_handle.create_dataset('test_set/features', data=features_test, compression='gzip')
	file_handle.create_dataset('test_set/labels', data=labels_test, compression='gzip')


print("Creation time:" , round(time()-t0, 3), "seconds")
