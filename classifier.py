import numpy as np
from PIL import Image, ImageOps
import h5py
import pickle
from nets import *
from create_dictionary import KutenDictionary


class Classifier():
	def __init__(self):
		self.model = model_M6_2()
		self.model.load_weights('models/model_M6_2.hdf5')
		with open('data/dictionary.pkl', 'rb') as f:
			self.lookup = pickle.load(f)

	# Method accepts ndarrays of shape (1,1,64,64)
	def classify(self, img_array):
		print(img_array)
		print(img_array.shape)
		print(np.max(img_array))
		result = self.model.predict_classes(img_array, batch_size=1)
		print(self.lookup[result.item(0)])



if __name__=='__main__':
	#test_image = "test_images/test.jpg"
	#test_image = "test_images/ka.png"
	#test_image = "test_images/hi.png"
	test_image = "test_images/ki.png"
	#test_image = "test_images/u.png"

	image = Image.open(test_image)
	image.thumbnail((64,64), Image.ANTIALIAS)
	image = ImageOps.fit(image, (64,64))
	image = ImageOps.invert(image)
	image.show()
	image = np.array(image).dot([0.2, 0.7, 0.1]).reshape([1, 1, image.size[0], image.size[1]]) / 255.
	Classifier().classify(image)

	






