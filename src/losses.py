import numpy as np
import torch
from pystct import isdct
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from math import exp

def SNR(cover, container, phase, transform, transform_constructor=None, on_phase=False):
	"""
	Computes SNR (Signal-to-Noise-Ratio)
	metric between cover and container signals.
	First, it computes i[transform] over the spectrograms.
	Transform can be either [cosine] or [fourier]

	" A local SNR of 30dB is effectively a clean signal. 
	Listeners will barely notice anything better than 20dB, 
	and intelligibility is still pretty good at 0dB SNR "
	> http://www1.icsi.berkeley.edu/Speech/faq/speechSNR.html
	"""
	if transform == 'cosine':
		cover_wav = isdct(cover.squeeze(0).squeeze(0).cpu().detach().numpy(), frame_step=62)
		noise_wav = isdct((container - cover).squeeze(0).squeeze(0).cpu().detach().numpy(), frame_step=62)
	elif (transform == 'fourier') and (transform_constructor is not None):
		if on_phase:
			cover_wav = transform_constructor.inverse(cover.squeeze(1), phase.squeeze(1)).cpu().data.numpy()[..., :]
			noise_wav = transform_constructor.inverse(cover.squeeze(1), (container - phase).squeeze(1)).cpu().data.numpy()[..., :]
		else:
			cover_wav = transform_constructor.inverse(cover.squeeze(1), phase.squeeze(1)).cpu().data.numpy()[..., :]
			noise_wav = transform_constructor.inverse((container - cover).squeeze(1), phase.squeeze(1)).cpu().data.numpy()[..., :]
	else: raise Exception('Transform not defined')
	
	signal = np.sum(np.abs(np.fft.fft(cover_wav)) ** 2) / len(np.fft.fft(cover_wav))
	noise = np.sum(np.abs(np.fft.fft(noise_wav))**2) / len(np.fft.fft(noise_wav))
	if noise <= 0.00001 or signal <= 0.00001: return 0
	return 10 * np.log10(signal / noise)

def PSNR(secret, revealed):
	"""
	Computes PSNR (Peak Signal to Noise Ratio)
	metric between secret and revealed signals.
	"""
	s, r = (secret * 255.0), (revealed * 255.0)

	mse = torch.mean((s - r) ** 2)
	if mse == 0: mse = torch.Tensor([10e-46])
	return 20 * torch.log10(255.0 / torch.sqrt(mse))

def gaussian(window_size, sigma):
	"""
	Courtesy of https://github.com/Po-Hsun-Su/pytorch-ssim
	Function for the computation of Differentiable structural similarity (SSIM) index.
	"""
	gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
	return gauss/gauss.sum()

def create_window(window_size, channel):
	"""
	Courtesy of https://github.com/Po-Hsun-Su/pytorch-ssim
	Function for the computation of Differentiable structural similarity (SSIM) index.
	"""
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
	return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
	"""
	Courtesy of https://github.com/Po-Hsun-Su/pytorch-ssim
	Function for the computation of Differentiable structural similarity (SSIM) index.
	"""
	mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
	mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1*mu2

	sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
	sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
	sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

	C1 = 0.01**2
	C2 = 0.03**2

	ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
	
	if size_average:
		return ssim_map.mean()
	else:
		return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
	"""
	Courtesy of https://github.com/Po-Hsun-Su/pytorch-ssim
	Function for the computation of Differentiable structural similarity (SSIM) index.
	"""
	def __init__(self, window_size = 11, size_average = True):
		super(SSIM, self).__init__()
		self.window_size = window_size
		self.size_average = size_average
		self.channel = 1
		self.window = create_window(window_size, self.channel)

	def forward(self, img1, img2):
		(_, channel, _, _) = img1.size()

		if channel == self.channel and self.window.data.type() == img1.data.type():
			window = self.window
		else:
			window = create_window(self.window_size, channel)
			
			if img1.is_cuda:
				window = window.cuda(img1.get_device())
			window = window.type_as(img1)
			
			self.window = window
			self.channel = channel


		return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
	"""
	Courtesy of https://github.com/Po-Hsun-Su/pytorch-ssim
	Function for the computation of Differentiable structural similarity (SSIM) index.
	"""
	(_, channel, _, _) = img1.size()
	window = create_window(window_size, channel)
	
	if img1.is_cuda:
		window = window.cuda(img1.get_device())
	window = window.type_as(img1)
	
	return _ssim(img1, img2, window, window_size, channel, size_average)

def StegoLoss(secret, cover, container, container_2x, revealed, beta, cover2=None, container2=None, container_2x2=None):
	"""
	Our custom StegoLoss function: a convex combination of two reconstruction 
	losses (image: [loss_secret], spectrogram: [loss_cover]) where both terms 
	are leveraged with the hyperparameter [beta].

	The optional DTW term may be added outside this function in the trainer script.
	"""
	# The ...2 data are the magnitude data to be used only with mag+phase

	assert (cover2 is None and container2 is None) or (cover2 is not None and container2 is not None)

	loss_cover = F.mse_loss(cover, container)
	print('Original cover:', loss_cover)
	if cover2 is not None:
		# Loss cover is adding MSEs for the magnitude and phase
		loss_cover += F.mse_loss(cover2, container2)
		print('Modified cover:', loss_cover)
	loss_secret = nn.L1Loss()
	loss_spectrum = F.mse_loss(container, container_2x)
	if container_2x2 is not None:
		# Also add to the loss spectrum in the mag+phase case
		loss_spectrum += F.mse_loss(container2, container_2x2)
	loss = (1 - beta) * (loss_cover) + beta * loss_secret(secret, revealed)
	print('Total loss:', loss)
	return loss, loss_cover, loss_secret(secret, revealed), loss_spectrum
