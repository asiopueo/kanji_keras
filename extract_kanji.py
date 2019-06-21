#!/usr/bin/python

from PIL import Image
import numpy as np
import sys
from time import time

import h5py


SIZE_X = 64
SIZE_Y = 63

SAMPLE_WIDTH = 512
TOTAL_RECORDS = 51200
# 51200/160 = 320 different Kanji in dataset
# TOTAL_RECORDS = 50560
DATA_FILE = "etlcdb/ETL8B/ETL8B2C1"
DB_FILE = "data/jisx0208.db"


class Extractor():

	def __init__(self):
		with open(DATA_FILE, 'rb') as file_handle:
			self.byte_buffer = file_handle.read( (TOTAL_RECORDS+1)*SAMPLE_WIDTH )



	def reader(self, counter, padding=True):
		data = self.byte_buffer[SAMPLE_WIDTH * counter : SAMPLE_WIDTH * (counter+1) ]
		data = data[8:].hex()

		tmp=[]
		for i in data:
			tmp.append(format(int(i,16),'04b'))

		tmp_str = ''.join(tmp)
		if padding:
			tmp_str = tmp_str + ''.join(['0']*SIZE_X)
		return tmp_str

	def getJIS(self, counter):
		data = self.byte_buffer[SAMPLE_WIDTH * counter : SAMPLE_WIDTH * (counter+1) ]
		JIS_code = data[2:4].hex()
		return JIS_code

	def getKuten(self, counter):
		data = self.byte_buffer[SAMPLE_WIDTH * counter : SAMPLE_WIDTH * (counter+1) ]
		JIS_code = data[2:4].hex()
		reading = data[4:8]
		JIS_code_ku = int(JIS_code[0:2], 16)-32
		JIS_code_ten = int(JIS_code[2:4], 16)-32
		return JIS_code_ku, JIS_code_ten


	def getArray(self, counter, padding=True):
		tmp_str = self.reader(counter, padding)
		array = np.eye(SIZE_Y+int(padding), SIZE_X).astype('uint8')

		for i in range(SIZE_Y+int(padding)):
			for j in range(SIZE_X):
				array[i][j] = 255 * int(tmp_str[i*SIZE_X+j])

		return array


	def addPadding(self, str):
		str.join('0' for i in range(SIZE_X))
		return str


	def show_on_console(self, tmp_str):
		output = ['']*SIZE_Y

		for row in range(SIZE_Y):
			for column in range(SIZE_X):
				output[row] = output[row] + tmp_str[row*SIZE_X+column]

		for row in range(SIZE_Y):
			print(output[row])




if __name__=='__main__':
	index = int(sys.argv[1])

	ext = Extractor()

	kanji_as_str = ext.reader(index)
	kanji_as_array = ext.getArray(index)

	# Output of a single Kanji:
	ext.show_on_console(kanji_as_str)

	jis_code = ext.getJIS(index)
	#print('JIS-code: {0}, Kuten-index: ({1}, {2})'.format(*getKuten(index)))

	from data.database import create_connection, getUnicode

	context = create_connection('./data/jisx0208.db')

	unicode_index = str(getUnicode(context, jis_code))

	character = chr(int(unicode_index, 16))
	print(character)


	#im = Image.fromarray(kanji_as_array, mode='L')
	#im.show()
	#im.save('example_output.jpg')
