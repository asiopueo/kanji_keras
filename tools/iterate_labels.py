import numpy as np
import h5py
from data.database import Database


with h5py.File('hdf5data/ETL8B2C1.hdf5', 'r') as f:
	t = f['test_set/labels'].value

	db = Database('./data/jisx0208.db')

	for count in range(0,*t.shape):
		#jis_code = hex(int(t[count]))
		jis_code = "{:04x}".format(int(t[count])) # Hex code as a string for DB query
		unic = db.getUnicode(jis_code)
		character = chr(int(unic,16))
		print(count, int(t[count]), jis_code, character)
