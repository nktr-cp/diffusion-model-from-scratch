import torch
from torch import nn

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

class UNet(nn.Module):
	def __init__(self, in_ch=1, time_embed_dim=100):
		super().__init__()
		self.time_embed_dim = time_embed_dim

		self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
		self.down2 = ConvBlock(64, 128, time_embed_dim)
		self.bot1 = ConvBlock(128, 256, time_embed_dim)
		self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
		self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
		self.out = nn.Conv2d(64, in_ch, 1)

		self.maxpool = nn.MaxPool2d(2)
		self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
	
	def forward(self, x, timesteps):
		v = pos_encoding(timesteps, self.time_embed_dim, x.device)

		x1 = self.down1(x, v)
		x = self.maxpool(x1)
		x2 = self.down2(x, v)
		x = self.maxpool(x2)

		x = self.bot1(x, v)

		x = self.upsample(x)
		x = torch.cat([x, x2], dim = 1)
		x = self.up2(x, v)
		x = self.upsample(x)
		x = torch.cat([x, x1], dim = 1)
		x = self.up1(x, v)
		x = self.out(x)
		return x

class Diffuser:
	def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
		self.num_timesteps = num_timesteps
		self.device = device
		self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
		self.alphas = 1 - self.betas
		self.alpha_bars = torch.cumpord(self.alphas, dim=0)
	
	def add_noise(self, x_0, t):
		T = self.num_timesteps
		assert (1 <= t <= T).all()

		alpha_bar = self.alpha_bars(t - 1)
		N = alpha_bar.size(0)
		alpha_bar = alpha_bar.view(N, 1, 1, 1)

		noise = torch.randn_like(x_0, device=self.device)
		x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
		return x_t, noise
	
	def denoise(self, model, x, t):
		T = self.num_timesteps
		assert(1 <= t <= T).all()

		alpha = self.alphas[t - 1]
		alpha_bar = self.alpha_bars[t - 1]
		alpha_bar_prev = self.alpha_bars[t - 2]

		N = alpha.size()
		alpha = alpha.view(N, 1, 1, 1)
		alpha_bar = alpha_bar.view(N, 1, 1, 1)
		alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

		model.eval()
		with torch.no_grad():
			eps = model(x, t)
			model.train()

			noise = torch.randn_like(x, device=self.device)
			noise[t == 1] = 0

			mu = (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps) / torch.sqrt(alpha)
			std = torch.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
			return mu + noise * std

if __name__ == "__main__":
	v = pos_encoding(torch.tensor([1, 2, 3]), 16)
	print(v.shape)
