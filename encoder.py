

import torch.nn as nn
import math
import torch



class encoder(nn.Module):

	def __init__(self):
		super(encoder, self).__init__()
		self.network = nn.Sequential(
			nn.Conv2d( 3, 64 , kernel_size=(3,3), stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d( 64, 64 , kernel_size=(3,3), stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),

			nn.Conv2d( 64, 128 , kernel_size=(3,3), stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d( 128, 128 , kernel_size=(3,3), stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),

			nn.Conv2d( 128, 256 , kernel_size=(3,3), stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d( 256, 256 , kernel_size=(3,3), stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d( 256, 256 , kernel_size=(3,3), stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),

			nn.Conv2d( 256, 512 , kernel_size=(3,3), stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d( 512, 512 , kernel_size=(3,3), stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d( 512, 512 , kernel_size=(3,3), stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=1, padding=(1,1)),

			nn.Conv2d( 512, 512 , kernel_size=(3,3), stride=1, padding=2, dilation=(2,2)),
			nn.ReLU(inplace=True),
			nn.Conv2d( 512, 512 , kernel_size=(3,3), stride=1, padding=2, dilation=(2,2)),
			nn.ReLU(inplace=True),
			nn.Conv2d( 512, 512 , kernel_size=(3,3), stride=1, padding=2, dilation=(2,2)),
			nn.ReLU(inplace=True),
		)


	def forward(self, input):

		return self.network(input)


	def _initialize_weights(self):
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

