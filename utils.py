
# https://github.com/jiecaoyu/pytorch_imagenet/blob/master/networks/model_list/alexnet.py

import torch.nn as nn

class LRN(nn.Module):
	def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
		super(LRN, self).__init__()
		self.ACROSS_CHANNELS = ACROSS_CHANNELS
		if ACROSS_CHANNELS:
			self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
					stride=1,
					padding=(int((local_size-1.0)/2), 0, 0))
		else:
			self.average=nn.AvgPool2d(kernel_size=local_size,
					stride=1,
					padding=int((local_size-1.0)/2))
		self.alpha = alpha
		self.beta = beta


	def forward(self, x):
		if self.ACROSS_CHANNELS:
			div = x.pow(2).unsqueeze(1)
			div = self.average(div).squeeze(1)
			div = div.mul(self.alpha).add(1.0).pow(self.beta)
		else:
			div = x.pow(2)
			div = self.average(div)
			div = div.mul(self.alpha).add(1.0).pow(self.beta)
		x = x.div(div)
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
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
