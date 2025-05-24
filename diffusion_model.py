import torch
from torch import nn

class ConvBlock(nn.Module):
	def __init__(self, in_ch, out_ch, time_embed_dim):
		super().__init__()
		self.convs = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(),
			nn.Conv2d(out_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(),
		)
		self.mlp = nn.Sequencial(
			nn.Linear(time_embed_dim, in_ch),
			nn.ReLU(),
			nn.Linear(in_ch, in_ch)
		)
	
	def forward(self, x, v):
		N, C, _, _ = x.shape
		v = self.mlp(v)
		v = v.view(N, C, 1, 1)
		return self.convs(x + v)

# Positional Encoding
def _pos_encoding(time: int, output_dim, device='cpu'):
	D = output_dim
	encoded_vector = torch.zeros(D, device=device)

	# [0, 1, ..., D-1]
	i = torch.arange(0, D, device=device)
	div_term = 10000 ** (i / D)

	# i: even
	encoded_vector[0::2] = torch.sin(time / div_term[0::2])
	# i: odd
	encoded_vector[1::2] = torch.cos(time / div_term[1::2])
	
	return encoded_vector

def pos_encoding(times: torch.tensor, output_dim, device='cpu'):
	batch_size = len(times)
	encoded_matrix = torch.zeros(batch_size, output_dim, device=device)

	for i in range(batch_size):
		encoded_matrix[i] = _pos_encoding(times[i], output_dim, device)

	return encoded_matrix

if __name__ == "__main__":
	v = pos_encoding(torch.tensor([1, 2, 3]), 16)
	print(v.shape)
