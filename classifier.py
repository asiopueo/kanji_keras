import numpy as np
from PIL import Image, ImageOps
import h5py
from nets import *
import ETL8B2C1
import data.database
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Classifier():
	def __init__(self):
		self.model = model_M6_2()
		self.model.load_weights('models/model_M6_2.hdf5')
		self.etldb = ETL8B2C1.Database('data/ETL8B2C1.db')
		self.jisdb = data.database.Database('data/jisx0208.db')

	# Method accepts ndarrays of shape (1,1,64,64)
	def classify(self, img_array):
		#result = self.model.predict(img_array, batch_size=1)
		result = self.model.predict_classes(img_array, batch_size=1)
		print(result)

		jis_code = self.etldb.getJIS(result.item(0)+1) # Check the table again!
		print("JIS-code: ", jis_code)
		print("Character: ", self.jisdb.getCharacter(jis_code))




if __name__=='__main__':
	test_image = "test_images/ka.png"
	#test_image = "test_images/hi.png"
	#test_image = "test_images/ki.png"
	#test_image = "test_images/u.png"
	#test_image = "example_output.png"

	image = Image.open(test_image)
	image.thumbnail((64,64), Image.ANTIALIAS)
	image = ImageOps.fit(image, (64,64))
	image = ImageOps.invert(image) # For pngs
	image.show()

	#image = np.array(image).reshape((1,1,64,64)) # For data set extracts
	image = np.array(image).dot([0.2, 0.7, 0.1]).reshape([1, 1, image.size[0], image.size[1]]) / 255. # For pngs
	print(image.shape)

	Classifier().classify(image)

	






