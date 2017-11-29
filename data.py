

import glob
import os
from scipy.misc import imread
from PIL import Image


class Salicon():
	def __init__(self, path='/media/ramin/monster/dataset/saliency/salicon/', sets=['train','val']):
		self.path = path
		self.images = {key : sorted(glob.glob(os.path.join(path, 'images/{0}/*.jpg'.format(key)))) for key in sets}
		self.maps = {key : sorted(glob.glob(os.path.join(path, 'maps/{0}/*.png'.format(key)))) for key in sets}

		self.pointer = {key: 0 for key in sets}
		for key in sets:
			assert len(self.images[key]) == len(self.maps[key])

	def __len__(self):
		return len(self.images['train'])

	def next_batch(self, batch_size=4, key='train'):
		batch = list()

		for i in range(batch_size):
			img = Image.open(self.images[key][self.pointer[key]])
			fix_map = Image.open(self.maps[key][self.pointer[key]])
			if img.mode != 'RGB':
				img = img.convert('RGB')

			assert img.size == fix_map.size

			batch.append([img, fix_map])

			self.pointer[key] +=1
			if self.pointer[key] >= len(self.images[key]):
				self.pointer[key] = 0

		return batch
