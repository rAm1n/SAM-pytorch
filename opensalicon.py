

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch
from utils import LRN
import os




class Saliency(nn.Module):

	def __init__(self, net_1, net_2=None):
		super(Saliency, self).__init__()

		self.net_1 = net_1
		self.net_2 = net_2

		#if self.net_2 is not None:
		#	self.classifier = nn.Sequential(
		#		nn.Conv2d(1024, 1, kernel_size=(1,1), stride=1)
		#	)
		#else:
		#	self.classifier = nn.Sequential(
		#		nn.Conv2d(512, 32, kernel_size=(1,1), stride=1),
		#		nn.Conv2d(32, 1, kernel_size=(1,1), stride=1)
		#	)

		self.classifier = nn.Sequential(
			nn.Conv2d( 1024, 1 , kernel_size=(1,1), stride=1),
	#		nn.Conv2d( 32, 1, kernel_size=(1,1), stride=1)
		)

		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.LogSoftmax(dim=-1)
		self.scale4 = nn.Upsample(scale_factor=4)
		self.scale2 = nn.Upsample(scale_factor=2)
		self.scale = nn.Upsample(size=(80,110))
		self._initialize_weights()

	def forward(self, input_1, input_2=None, extracted_layers=['23','30']):
		x = input_1
		outputs = []
        	for name, module in self.net_1._modules.items():
			x = module(x)
			if name in extracted_layers:
				outputs += [x]

		merge = torch.cat([outputs[0], self.scale2(outputs[1])], dim=1)
		x = self.scale(merge)
		#x = self.net_1(input_1)
		#x = self.scale(x)
		#if self.net_2 is not None:
		#	if input_2 is not None:
		#		op_2 = self.net_2(input_2)
		#		op_2 = self.scale(op_2)
		#		x = torch.cat([x , op_2], dim=1)


		#return self.classifier(x)
		# x = self.up4(x)
		x = self.classifier(x)
		b , c, h, w = x.size()
		return self.sigmoid(x.view(b,-1)).view(b,c,h,w)
		# return self.softmax(x.view(b,-1)).view(b,c,h,w)

	def _initialize_weights(self,vgg=False):
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()
		if vgg:
			self.load_vgg_weights()

	def load_vgg_weights(self, url='https://download.pytorch.org/models/vgg16-397923af.pth'):
		vgg = model_zoo.load_url(url)
		for idx,m in enumerate(self.net_1.modules()):
			idx = idx - 1 
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data = vgg.get('features.{0}.weight'.format(idx))
				if m.bias is not None:  
					m.bias.data = vgg.get('features.{0}.bias'.format(idx))
		if self.net_2 is not None:
			for idx,m in enumerate(self.net_2.modules()):
				idx = idx - 1 
				if isinstance(m, nn.Conv2d):
					n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
					m.weight.data = vgg.get('features.{0}.weight'.format(idx))
					if m.bias is not None:  
						m.bias.data = vgg.get('features.{0}.bias'.format(idx))


	def save_checkpoint(self, state, ep, step, max_keep=5, path='/media/ramin/monster/models/deep_shallow/'):
		filename = os.path.join(path, 'ck-{0}-{1}.pth.tar'.format(ep, step))
		torch.save(state, os.path.join(path, 'ck-last.path.tar'))
		torch.save(state, filename)
		def sorted_ls(path):
			mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
			return list(sorted(os.listdir(path), key=mtime))
		files = sorted_ls(path)[:-max_keep]
		for item in files:
			os.remove(os.path.join(path, item))

	def load_checkpoint(self, path='/media/ramin/monster/models/deep_shallow/', filename=None):
		if not filename:
			filename = os.path.join(path, 'ck-last.path.tar')
		else:
			filename = os.path.join(path, filename)

		self.load_state_dict(torch.load(filename))






def saliency(pretrained=True, dual=True, cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], **kwargs):
	"""VGG 16-layer model (configuration "D")

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	def make_layers(cfg, batch_norm=False):
		layers = []
		in_channels = 3
		for v in cfg:
			if v == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
				if batch_norm:
					layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
				else:
					layers += [conv2d, nn.ReLU(inplace=True)]
				in_channels = v
		return nn.Sequential(*layers)

	if dual:
		model = Saliency(make_layers(cfg), make_layers(cfg), **kwargs)
	else:
		model = Saliency(make_layers(cfg), **kwargs)
	if pretrained:
		model.load_vgg_weights()
	return model





