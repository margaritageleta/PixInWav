import torch
from torch import utils
import torch.nn as nn
import torch.nn.functional as F

class PrepNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.features3x3 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(4, 1), padding=(1, 1)),
			nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
			nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		)

		self.features4x4 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(4, 4), stride=(8, 2), padding=(1, 1)),
			nn.Conv2d(16, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
		)

		self.features5x5 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(8, 2), padding=(1, 2)),
			nn.Conv2d(16, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
		)

		self.deep_features3x3 = nn.Sequential(
			nn.Conv2d(48, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		)

		self.deep_features4x4 = nn.Sequential(
			nn.Conv2d(48, 16, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
			nn.Conv2d(16, 4, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2)),
		)

		self.deep_features5x5 = nn.Sequential(
			nn.Conv2d(48, 4, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
		)

	def forward(self, x):
		features3x3 = self.features3x3(x)
		features4x4 = self.features4x4(x)
		features5x5 = self.features5x5(x)

		x = torch.cat((features3x3, features4x4, features5x5), 1)

		deep_features3x3 = self.deep_features3x3(x)
		deep_features4x4 = self.deep_features4x4(x)
		deep_features5x5 = self.deep_features5x5(x)

		x = torch.cat((deep_features3x3, deep_features4x4, deep_features5x5), 1)
		return x


class HidingNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.features3x3 = nn.Sequential(
			nn.Conv2d(15, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		)

		self.features4x4 = nn.Sequential(
			nn.Conv2d(15, 32, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
			nn.Conv2d(32, 32, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2))
		)

		self.features5x5 = nn.Sequential(
			nn.Conv2d(15, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
			nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
		)

		self.deep_features3x3 = nn.Sequential(
			nn.ConvTranspose2d(96, 16, kernel_size=(3, 3), stride=(4, 2), padding=(1, 1), output_padding=(3, 1)),
			nn.ConvTranspose2d(16, 4, kernel_size=(3, 3), stride=(4, 2), padding=(1, 1), output_padding=(3, 1)),
		)

		self.deep_features4x4 = nn.Sequential(
			nn.ConvTranspose2d(96, 16, kernel_size=(4, 4), stride=(4, 2), padding=(1, 1), output_padding=(2, 0)),
			nn.ConvTranspose2d(16, 4, kernel_size=(4, 4), stride=(4, 2), padding=(1, 1), output_padding=(2, 0)),
		)

		self.deep_features5x5 = nn.Sequential(
			nn.ConvTranspose2d(96, 16, kernel_size=(5, 5), stride=(4, 2), padding=(1, 2), output_padding=(1, 1)),
			nn.ConvTranspose2d(16, 4, kernel_size=(5, 5), stride=(4, 2), padding=(1, 2), output_padding=(1, 1)),
		)

		self.funnel = nn.Sequential(
			nn.Conv2d(12, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		)

	def forward(self, x):
		print(x.shape)
		features3x3 = self.features3x3(x)
		features4x4 = self.features4x4(x)
		features5x5 = self.features5x5(x)

		x = torch.cat((features3x3, features4x4, features5x5), 1)

		deep_features3x3 = self.deep_features3x3(x)
		deep_features4x4 = self.deep_features4x4(x)
		deep_features5x5 = self.deep_features5x5(x)

		x = torch.cat((deep_features3x3, deep_features4x4, deep_features5x5), 1)
		x = self.funnel(x)

		return x


class RevealNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.features3x3 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(4, 1), padding=(1, 1)),
			nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
			nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		)

		self.features4x4 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(4, 4), stride=(8, 2), padding=(1, 1)),
			nn.Conv2d(16, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
		)

		self.features5x5 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(8, 2), padding=(1, 2)),
			nn.Conv2d(16, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
		)

		self.deep_features3x3 = nn.Sequential(
			nn.Conv2d(48, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		)

		self.deep_features4x4 = nn.Sequential(
			nn.Conv2d(48, 16, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
			nn.Conv2d(16, 4, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2)),
		)

		self.deep_features5x5 = nn.Sequential(
			nn.Conv2d(48, 4, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
		)

		self.funnel = nn.Sequential(
			nn.Conv2d(12, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
		)

	def forward(self, x):
		features3x3 = self.features3x3(x)
		features4x4 = self.features4x4(x)
		features5x5 = self.features5x5(x)

		x = torch.cat((features3x3, features4x4, features5x5), 1)

		deep_features3x3 = self.deep_features3x3(x)
		deep_features4x4 = self.deep_features4x4(x)
		deep_features5x5 = self.deep_features5x5(x)

		x = torch.cat((deep_features3x3, deep_features4x4, deep_features5x5), 1)
		x = self.funnel(x)
		return x


class StegoNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.PN = PrepNet()
		self.HN = HidingNet()
		self.RN = RevealNet()

	def forward(self, secret, cover):
		cover_features = self.PN(cover)
		concat = torch.cat((cover_features, secret), 1)
		container = self.HN(concat)
		revealed = self.RN(container)

		return container, revealed