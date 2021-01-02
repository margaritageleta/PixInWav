import torch
from torch import utils
import torch.nn as nn
import torch.nn.functional as F


class PrepNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.features3x3 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(4, 1), padding=(1, 1)),
			nn.LeakyReLU(0.8),
			nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
			nn.LeakyReLU(0.8),
			nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
		)

		self.features5x5 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(8, 2), padding=(1, 2)),
			nn.LeakyReLU(0.8),
			nn.Conv2d(16, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
		)

		self.deep_features3x3 = nn.Sequential(
			nn.Conv2d(32, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		)

		self.deep_features5x5 = nn.Sequential(
			nn.Conv2d(32, 4, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
		)

	def forward(self, x):
		features3x3 = self.features3x3(x)
		features5x5 = self.features5x5(x)

		x = torch.cat((features3x3, features5x5), 1)

		deep_features3x3 = self.deep_features3x3(x)
		deep_features5x5 = self.deep_features5x5(x)

		x = torch.cat((deep_features3x3, deep_features5x5), 1)
		return x


class HidingNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.features3x3 = nn.Sequential(
			nn.Conv2d(24, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.LeakyReLU(0.8),
			nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.LeakyReLU(0.8),
			nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		)

		self.features5x5 = nn.Sequential(
			nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
			nn.LeakyReLU(0.8),
			nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
		)

		self.deep_features3x3 = nn.Sequential(
			nn.Upsample(scale_factor=(4, 2), mode='bilinear'),
			nn.Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.LeakyReLU(0.8),
			nn.Upsample(scale_factor=(4, 2), mode='bilinear'),
			nn.Conv2d(16, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		)

		self.deep_features5x5 = nn.Sequential(
			nn.Upsample(scale_factor=(4, 2), mode='bilinear'),
			nn.Conv2d(64, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
			nn.LeakyReLU(0.8),
			nn.Upsample(scale_factor=(4, 2), mode='bilinear'),
			nn.Conv2d(16, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		)

		self.funnel = nn.Sequential(
			nn.Conv2d(8, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		)

	def forward(self, x):
		features3x3 = self.features3x3(x)
		features5x5 = self.features5x5(x)

		x = torch.cat((features3x3, features5x5), 1)

		deep_features3x3 = self.deep_features3x3(x)
		deep_features5x5 = self.deep_features5x5(x)

		x = torch.cat((deep_features3x3, deep_features5x5), 1)
		x = self.funnel(x)

		return x


class RevealNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.features3x3 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(4, 1), padding=(1, 1)),
			nn.LeakyReLU(0.8),
			nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
			nn.LeakyReLU(0.8),
			nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
		)

		self.features5x5 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(8, 2), padding=(1, 2)),
			nn.LeakyReLU(0.8),
			nn.Conv2d(16, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
		)

		self.deep_features3x3 = nn.Sequential(
			nn.Conv2d(32, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		)

		self.deep_features5x5 = nn.Sequential(
			nn.Conv2d(32, 4, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
		)

		self.funnel = nn.Sequential(
			nn.Conv2d(8, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		)

	def forward(self, x):
		features3x3 = self.features3x3(x)
		features5x5 = self.features5x5(x)

		x = torch.cat((features3x3, features5x5), 1)

		deep_features3x3 = self.deep_features3x3(x)
		deep_features5x5 = self.deep_features5x5(x)

		x = torch.cat((deep_features3x3, deep_features5x5), 1)
		x = self.funnel(x)
		return x


class StegoNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.PN = PrepNet()
		self.HN = HidingNet()
		self.RN = RevealNet()

		self.features3x3 = nn.Sequential(
			nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.LeakyReLU(0.8),
			nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.LeakyReLU(0.8),
			nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		)

		self.features5x5 = nn.Sequential(
			nn.Conv2d(3, 8, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
			nn.LeakyReLU(0.8),
			nn.Conv2d(8, 8, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
		)

	def forward(self, secret, cover):
		# print('Process Network working ...')
		cover_features = self.PN(cover)
		# print(cover_features.shape)

		features3x3 = self.features3x3(secret)
		features5x5 = self.features5x5(secret)
		secret_features = torch.cat((features3x3, features5x5), 1)
		# print(secret_features.shape)

		concat = torch.cat((cover_features, secret_features), 1)
		# print(concat.shape)
		# print('Hiding Network working ...')
		container = self.HN(concat)
		# print('Reveal Network working ...')
		revealed = self.RN(container)

		return container, revealed