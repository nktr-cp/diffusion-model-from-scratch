import torch
from torch import nn

class ConvBlock(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.convs = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(),
			nn.Conv2d(out_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(),
		)
	def forward(self, x):
		return self.convs(x)
