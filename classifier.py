import numpy as np
from PIL import Image, ImageOps
import h5py
import pickle
from nets import *



#test_image = "test_images/test.jpg"
#test_image = "test_images/ka.png"
test_image = "test_images/ki.png"
#test_image = "test_images/u.png"

model = model_M6_1()
model.load_weights('models/model_M6_1.hdf5')


image = Image.open(test_image)
image.thumbnail((64,64), Image.ANTIALIAS)
image = ImageOps.fit(image, (64,64))
image = ImageOps.invert(image)
#image.show()

data = np.array(image).dot([0.2, 0.7, 0.1]).reshape([1, 1, image.size[0], image.size[1]]) / 255.

#print data.shape
#print np.max(data)
result = model.predict_classes(data, batch_size=1)



with open('data/dictionary.pkl', 'rb') as f:
	from create_dictionary import KutenDictionary
	lookup = pickle.load(f)
	print(lookup[result.item(0)])
	print("蓮")






