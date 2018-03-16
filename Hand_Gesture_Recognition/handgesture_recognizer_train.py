from sklearn.utils import shuffle

import cv2
import numpy as np
import matplotlib.pyplot as plt


class TrainHandGestureClasiffier():
	def __init__(self,trainfile_path='Marcel-Train'):
		self.DATASET_PATH = trainfile_path

	def read_dataset(self):

		classes = [folder for folder in sorted(os.listdir(self.DATASET_PATH))]

		images = []
		for c in classes:
			images += ([os.path.join(self.DATASET_PATH, c, path) for path in os.listdir(os.path.join(self.DATASET_PATH, c))])

		images = shuffle(images, random_state=42)

		#we want to use a 15% validation split
		vsplit = int(len(images) * 0.15)
		train = images[:-vsplit]
		val = images[-vsplit:]

		#show some stats
		print "CLASS LABELS:", classes
		print "TRAINING IMAGES:", len(train)
		print "VALIDATION IMAGES:", len(val)

		return classes, train, val
	
	
if __name__ == '__main__':
	trainhandgestureclasiffier= TrainHandGestureClasiffier()
	CLASSES, TRAIN, VAL  = trainhandgestureclasiffier.read_dataset()
	
	#print VAL
	#messageclassifier.train_SVMClassifier() 
