
import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from opensalicon import saliency
import numpy as np
import cv2
import time
from PIL import Image
import torchvision.transforms as transforms
from data import Salicon
import os
from scipy.ndimage.filters import gaussian_filter

batch_size=4

# im_size=(480,480)
# final_size=(480,480)
# opt='SGD'
# lr=1.30208333333e-07
# B1=0.01
# B2=0.999
# decay= 5e-07
# momentum=0.9
# step_size=5
# gamma= 0.5
# eps=1e-5
# epoch = 5

im_size_1=(720, 960)
im_size_2=(270,360)
final_size=(640,480) ##################3 PIL size reverse
opt='ADAM'
#lr=0.000001
lr=1e-5
B1=0.01
B2=0.999
decay=0.0005
momentum=0.9
step_size=5
gamma= 0.1
eps=1e-5
epoch = 15
fine_tune=False

eps = 0.00005


ck_path = '/media/ramin/monster/models/deep-shallow-2/'
res_path = 'result-2/'

dtype = torch.cuda.FloatTensor
torch.set_default_tensor_type('torch.cuda.FloatTensor')


img_processor_1 = transforms.Compose([
	transforms.Resize(im_size_1),
		transforms.ToTensor(),
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
			# std = [1.0,1.0,1.0]
		)
	]
)


img_processor_2 = transforms.Compose([
	transforms.Resize(im_size_2),
		transforms.ToTensor(),
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
			# std = [1.0,1.0,1.0]
		)
	]
)

map_processor = transforms.Compose([
	transforms.Resize((80,110)),
		transforms.ToTensor(),
   #      transforms.Normalize(
   #          mean=[0.5 ],
			# std=[1]
   #      ), 
	]
)







def train():

	global crit, output, fix_maps, images, target, batch, d, fix, pred

	print('building model')
	model = saliency(dual=False)
	print('initializing')
	model._initialize_weights(vgg=True)
	model = model.cuda()



	print('setting up the objective function')
	# crit = nn.KLDivLoss()
	crit = nn.MSELoss()
	# crit = nn.MultiLabelSoftMarginLoss()


	print('setting up optimizer')
	if opt == 'ADAM':
		optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(B1, B2), eps=eps)
	elif opt=='SGD':
		optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=decay, momentum=momentum)
	elif opt=='adadelta':
		optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=decay)

	print('fina_tune : {0}'.format(fine_tune))
	if  fine_tune:
		for param in model.net_1.parameters():
				param.requires_grad = False
		if model.net_2 is not None:
			for param in model.net_2.parameters():
					param.requires_grad = False
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma )


	print('Loading dataset')
	d = Salicon()



	start = time.time()
	for ep in range(epoch):
		l = list()
		iteration = len(d) // batch_size
		for step in range(iteration):

			batch = d.next_batch(batch_size)

			# images_1 = [img_processor_1(img[0]) for img in batch]
			# images_1 = torch.stack(images_1) 

			images_2 = [img_processor_2(img[0]) for img in batch]
			images_2 = torch.stack(images_2) 

			
			fix_maps = [map_processor(fix[1]) for fix in batch]
			#fix_maps = list()
			#for fix in batch:
				#fix = map_processor(fix[1])
				#fix_maps.append(fix / fix.sum())
			fix_maps = torch.stack(fix_maps)#.view(batch_size, -1)

			

			# images_1 = Variable(images_1.cuda(), requires_grad=True)
			images_2 = Variable(images_2.cuda(), requires_grad=True)
			target = Variable(fix_maps.cuda(), requires_grad=False)


			# output = model(images_1, images_2)
			output = model(images_2)
			loss = crit(output, target)

			try:
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				l.append(loss.data[0])
				#torch.cuda.synchronize()
			except Exception as x:
				print(x)
				return 0


			# if step%20 == 0 and step != 0:
			# 	print('epoch {0} , step {1} / {2}   , train-loss: {3}  ,  time: {4}'.format(ep, step, iteration, np.array(l).mean(), time.time()-start))
			# 	l = list()
			# 	start = time.time()

			if step%100 == 0 and step !=0:
				try:
					model.eval()

					batch = d.next_batch(batch_size, key='val') 

					images_1 = [img_processor_1(img[0]) for img in batch]
					images_1 = torch.stack(images_1) 


					images_2 = [img_processor_2(img[0]) for img in batch]
					images_2 = torch.stack(images_2) 


					# fix_maps = [map_processor(fix[1]) for fix in batch]
					# fix_maps = torch.stack(fix_maps)#.view(batch_size, -1)

					fix_maps = list()
					for fix in batch:
						fix = map_processor(fix[1]) 
						# fix_maps.append(fix / fix.sum())
						fix_maps.append(fix)
					fix_maps = torch.stack(fix_maps)


					images_1 = Variable(images_1.cuda(), requires_grad=True)
					images_2 = Variable(images_2.cuda(), requires_grad=True)
					target = Variable(fix_maps.cuda(), requires_grad=False)

					output = model(images_2)
					loss = crit(output, target)
					

					output = output.view(batch_size, 80, 110).data.cpu().numpy()
					# output = output.permute(0,2,3,1).squeeze().data.cpu().numpy()
					images_1 = images_1.permute(0,2,3,1).data.cpu().numpy()
					fix_maps = fix_maps.cpu().numpy()

					for idx in range(batch_size):
						pred = output[idx]
						# pred -= pred.mean()
						#pred = (pred - pred.min()) / (pred.max() - pred.min())
						#pred = pred - pred.min()
						#pred = pred / pred.max()
						pred *= 255.0
						#pred = np.minimum(pred, 255)
						pred = gaussian_filter(pred, sigma=3)
						pred = pred.astype(np.uint8)
						pred = Image.fromarray(pred).resize(final_size)
#						mask = mask.convert('RGB')

						img = images_1[idx]
						img = (img - img.min()) / (img.max() - img.min())
						img *= 255.0
						img = img.astype(np.uint8)
						img =  Image.fromarray(img).resize(final_size)


						gt = batch[idx][1].resize(final_size)

						pred.save(os.path.join(res_path, '{0}-{1}-{2}-pred.png'.format(ep, step, idx)))
						gt.save(os.path.join(res_path, '{0}-{1}-{2}-gt.png'.format(ep, step, idx)))
						img.save(os.path.join(res_path, '{0}-{1}-{2}-img.png'.format(ep, step, idx)))
	
						# pred = pred.convert('RGB')
						# new_im = Image.blend(img, pred, alpha=0.9).convert('RGB')
						# new_im.save(os.path.join(res_path, '{0}-{1}-{2}.png'.format(ep, step, idx)))

					print('epoch {0} , step {1} / {2}   , train-loss: {3} , val-loss: {4}, time : {5}'.format(ep, step, iteration, np.array(l).mean(), loss.data[0], time.time()-start))

					model.train()
					l = list()
					start = time.time()
					del loss, images_1, images_2, target, output

				except Exception as x:
					print(x)


			if step%2000 == 0:
				scheduler.step()
		model.save_checkpoint(model.state_dict(), ep, step, path=ck_path)
	

train()

