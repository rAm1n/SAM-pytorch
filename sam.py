import torch.nn as nn
import math
import torch

from encoder import encoder
from convlstm import *
import pickle



class SAM(nn.Module):

	def __init__(self):
		super(SAM, self).__init__()

		self.encoder = encoder()
		self.attn_conv_lstm = ConvLSTM((31,41), 512, 512, (3,3), 1, True)
		self.conv_out = nn.Sequential(
			nn.Conv2d( 512, 64 , kernel_size=(4,4), stride=1),
			nn.ReLU(inplace=True),
			nn.Conv2d( 64, 1 , kernel_size=(1,1), stride=1), dilation=(4,4),
			nn.ReLU(inplace=True),
			)

		self.scale = nn.Upsample(size=(240,320))


	def forward(self, input):
		x = self.encoder(input)
		h_ , c_ = self.attn_conv_lstm(x, itr=5)

		out = self.conv_out(h_[:,-1])


	def _initialize_weights(self, pre_trained=True):
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
		if pre_trained:
			self._load_sam_weights()

	def _load_sam_weights(self, filename='weights.pkl'):
		w = pickle.load(open(filename, 'r'))



