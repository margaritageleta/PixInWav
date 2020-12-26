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
			nn.LeakyReLU(0.8),
		)

		self.features4x4 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(4, 4), stride=(8, 2), padding=(1, 1)),
			nn.LeakyReLU(0.8),
			nn.Conv2d(16, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
			nn.LeakyReLU(0.8),
		)

		self.features5x5 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(8, 2), padding=(1, 2)),
			nn.LeakyReLU(0.8),
			nn.Conv2d(16, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
			nn.LeakyReLU(0.8),
		)

		self.deep_features3x3 = nn.Sequential(
			nn.Conv2d(48, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.LeakyReLU(0.8),
		)

		self.deep_features4x4 = nn.Sequential(
			nn.Conv2d(48, 16, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
			nn.LeakyReLU(0.8),
			nn.Conv2d(16, 4, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2)),
			nn.LeakyReLU(0.8),
		)

		self.deep_features5x5 = nn.Sequential(
			nn.Conv2d(48, 4, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
			nn.LeakyReLU(0.8),
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
            nn.LeakyReLU(0.8),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.8),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.8),
        )

        self.features4x4 = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.8),
            nn.Conv2d(32, 32, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2)),
            nn.LeakyReLU(0.8),
        )

        self.features5x5 = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
			nn.LeakyReLU(0.8),
            nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.LeakyReLU(0.8),
        )

        self.deep_features3x3 = nn.Sequential(
            nn.Upsample(scale_factor=(4,2), mode='bilinear'),
            nn.Conv2d(96, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.8),
            nn.Upsample(scale_factor=(4,2), mode='bilinear'),
        )

        self.deep_features4x4 = nn.Sequential(
            nn.Upsample(scale_factor=(16,4), mode='bilinear'),
            nn.Conv2d(96, 16, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.8),
            nn.Conv2d(16, 4, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2)),
            nn.LeakyReLU(0.8),
        )

        self.deep_features5x5 = nn.Sequential(
            nn.Upsample(scale_factor=(4,2), mode='bilinear'),
            nn.Conv2d(96, 4, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.LeakyReLU(0.8),
            nn.Upsample(scale_factor=(4,2), mode='bilinear'),
        )

        self.funnel = nn.Sequential(
            nn.Conv2d(12, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.8),
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


class RevealNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.features3x3 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(4, 1), padding=(1, 1)),
			nn.LeakyReLU(0.4),
			nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
			nn.LeakyReLU(0.4),
			nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
			nn.LeakyReLU(0.4),
		)

		self.features4x4 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(4, 4), stride=(8, 2), padding=(1, 1)),
			nn.LeakyReLU(0.4),
			nn.Conv2d(16, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
			nn.LeakyReLU(0.4),
		)

		self.features5x5 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(8, 2), padding=(1, 2)),
			nn.LeakyReLU(0.4),
			nn.Conv2d(16, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
			nn.LeakyReLU(0.4),
		)

		self.deep_features3x3 = nn.Sequential(
			nn.Conv2d(48, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
			nn.LeakyReLU(0.4),
		)

		self.deep_features4x4 = nn.Sequential(
			nn.Conv2d(48, 16, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
			nn.LeakyReLU(0.4),
			nn.Conv2d(16, 4, kernel_size=(4, 4), stride=(1, 1), padding=(2, 2)),
			nn.LeakyReLU(0.4),
		)

		self.deep_features5x5 = nn.Sequential(
			nn.Conv2d(48, 4, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
			nn.LeakyReLU(0.4),
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
		# print('Process Network working ...')
		cover_features = self.PN(cover)
		concat = torch.cat((cover_features, secret), 1)
		# print('Hiding Network working ...')
		container = self.HN(concat)
		# print('Reveal Network working ...')
		revealed = self.RN(container)

		return container, revealed