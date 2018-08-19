#!/usr/bin/python

import pickle

TOTAL_RECORDS = 51200
SAMPLE_WIDTH = 512
DATA_FILE = "etlcdb/ETL8B/ETL8B2C1"




class KutenDictionary(dict):
	def __setitem__(self, key, value):
		if key in self:
			del self[key]
		if value in self:
			del self[value]
		dict.__setitem__(self, key, value)
		dict.__setitem__(self, value, key)

	def __delitem__(self, key):
		dict.__delitem__(self, self[key])
		dict.__delitem__(self, key)

	def __len__(self):
		return dict.__len__(self) / 2


def reader(counter):
	data = byte_buffer[SAMPLE_WIDTH * counter : SAMPLE_WIDTH * (counter+1) ]
	JIS_code = data[2:4]
	reading = data[4:8]
	JIS_code_ku = JIS_code[0] - 32
	JIS_code_ten = JIS_code[1] - 32
	return (JIS_code_ku, JIS_code_ten)



with open(DATA_FILE, 'rb') as file_handle:
	byte_buffer = file_handle.read( (TOTAL_RECORDS+1) * SAMPLE_WIDTH )



if __name__ == '__main__':
	print("Creating KuTen-dictionary...")
	dictionary = KutenDictionary()

	for counter in range(0,TOTAL_RECORDS,160):
		dictionary[counter/160] = reader(counter+1)

	with open('data/dictionary.pkl', 'wb') as f:
	    pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)

	#print(dictionary[7])





