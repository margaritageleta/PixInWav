import os
import gc
import time
import torch
import wandb
import argparse
import numpy as np
import torch.nn as nn
from loader import loader
import torch.optim as optim
from umodel_rgb_shuffle import StegoUNet
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pystct import sdct_torch, isdct_torch
from losses import ssim, SNR, PSNR, StegoLoss
from pydtw import SoftDTW

def parse_keyword(keyword):
    if isinstance(keyword, bool):
       return keyword
    if keyword.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif keyword.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Wrong keyword.')

parser = argparse.ArgumentParser()
parser.add_argument('--beta', 
						type=float, 
						default=0.25, 
						metavar='DOUBLE',
						help='Beta hyperparameter'
					)
parser.add_argument('--lr', 
						type=float, 
						default=0.001, 
						metavar='DOUBLE',
						help='Learning rate hyperparameter'
					)
parser.add_argument('--experiment', 
						type=int, 
						default=0, 
						metavar='INT',
						help='Number of experiment'
					)
parser.add_argument('--summary', 
						type=str, 
						default=None, 
						metavar='STRING',
						help='Summary to be shown in wandb'
					)
parser.add_argument('--add_noise', 
						type=parse_keyword, 
						default=False, 
						metavar='BOOL',
						help='Boolean to add noise'
					)
parser.add_argument('--noise_kind', 
						type=str, 
						default=None, 
						metavar='STRING',
						help='Noise kind (gaussian, speckle, salt, pepper, salt&pepper)'
					)
parser.add_argument('--noise_amplitude', 
						type=float, 
						default=None, 
						metavar='FLOAT',
						help='Noise amplitude'
					)
parser.add_argument('--add_dtw_term', 
						type=parse_keyword, 
						default=False, 
						metavar='BOOL',
						help='Add DTW term in the loss function'
					)
parser.add_argument('--rgb', 
						type=parse_keyword, 
						default=True, 
						metavar='BOOL',
						help='Use RGB images or B&W'
					)
parser.add_argument('--from_checkpoint', 
						type=parse_keyword, 
						default=False, 
						metavar='BOOL',
						help='Use checkpoint listed by experiment number'
					)

# assert(True == False)

def save_checkpoint(state, is_best, filename=os.path.join(os.environ.get('OUT_PATH'),'models/checkpoint.pt')):
	 """Save checkpoint if a new best is achieved"""
	 if is_best:
		 print ("=> Saving a new best model")
		 print(f'SAVING TO: {filename}')
		 torch.save(state, filename)  # save checkpoint
	 else:
		 print ("=> Loss did not improve")

def compare_images(s, r):
	s = s.permute(0,2,3,1).detach().numpy().squeeze(0)
	r = r.permute(0,2,3,1).detach().numpy().squeeze(0)

	fig, ax = plt.subplots(1, 2, figsize=(10, 10))
	ax[0].imshow(s)
	ax[1].imshow(r)
	ax[0].set_title('Secret image')
	ax[1].set_title('Revealed image')
	ax[0].axis('off')
	ax[1].axis('off')
	plt.close('all')

	return fig

def viz2paper(s, r, cv, ct, log=True):
	s = s.permute(0,2,3,1).detach().numpy().squeeze(0)
	r = r.permute(0,2,3,1).detach().numpy().squeeze(0)
	cv = cv.detach().numpy().squeeze(0).squeeze(0)
	ct = ct.detach().numpy().squeeze(0).squeeze(0)
	
	s = (s * 255.0).astype(np.uint8)
	r = np.clip(r * 255.0, 0, 255).astype(np.uint8)

	fig, ax = plt.subplots(2, 2, figsize=(12, 10))
	ax[0,0].imshow(s)
	ax[1,0].imshow(r)
	ax[0,0].set_title('Secret image')
	ax[1,0].set_title('Revealed image')
	ax[0,0].axis('off')
	ax[1,0].axis('off')
	
	if log:
		img1 = ax[0,1].imshow(np.log(np.abs(cv)[:,] + 1), origin = 'upper', aspect = 'auto', cmap=plt.cm.get_cmap("jet"))
		ax[0,1].set_title('Cover STDCT log spectrogram')
		img2 = ax[1,1].imshow(np.log(np.abs(ct)[:,] + 1), origin = 'upper', aspect = 'auto', cmap=plt.cm.get_cmap("jet"))
		ax[1,1].set_title('Container STDCT log spectrogram')
	else:
		img1 = ax[0,1].imshow(np.abs(cv) [:,], origin = 'upper', aspect = 'auto', cmap=plt.cm.get_cmap("jet"))
		ax[0,1].set_title('Cover STDCT spectrogram')
		img2 = ax[1,1].imshow(np.abs(ct)[:,], origin = 'upper', aspect = 'auto', cmap=plt.cm.get_cmap("jet"))
		ax[1,1].set_title('Container STDCT spectrogram')
	
	ax[0,1].set_xlabel('Time [n]')
	ax[0,1].set_ylabel('Frequency')
	ax[1,1].set_xlabel('Time [n]')
	ax[1,1].set_ylabel('Frequency')
	
	plt.colorbar(img1, ax=ax[0,1])
	plt.colorbar(img2, ax=ax[1,1])
	plt.close('all')
	
	return fig


def train(model, tr_loader, vd_loader, beta, lr, epochs=5, prev_epoch = None, prev_i = None, summary=None, slide=50, experiment=0, add_dtw_term=False):

	wandb.init(project='PixInWavRGB')
	if summary is not None:
		wandb.run.name = summary
		wandb.run.save()
	wandb.watch(model)

	device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'Using device: {device}')

	if torch.cuda.device_count() > 1:
  		print("Let's use", torch.cuda.device_count(), "GPUs!")
  		model = nn.DataParallel(model)

	model.to(device)

	# Set to training mode
	model.train()

	# This is the number of parameters used in the model
	num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f'Number of model parameters: {num_params}')

	optimizer = optim.Adam(model.parameters(), lr=lr)

	ini = time.time()
	best_loss = np.inf

	softDTW = SoftDTW(gamma=1.0, normalize=True) 

	for epoch in range(epochs):

		if prev_epoch != None and epoch < prev_epoch - 1: continue

		train_loss, train_loss_cover, train_loss_secret, train_loss_spectrum, snr, psnr, ssim_secret, train_dtw_loss = [], [], [], [], [], [], [], []
		vd_loss, vd_loss_cover, vd_loss_secret, vd_snr, vd_psnr, vd_ssim, vd_dtw = [], [], [], [], [], [], []
		
		for i, data in enumerate(tr_loader):

			if prev_i != None and i < prev_i - 1: continue

			secrets, covers = data[0].to(device), data[1].to(device)
			secrets = secrets.permute(0, 3, 1, 2).type(torch.cuda.FloatTensor)
			covers = covers.unsqueeze(1)

			optimizer.zero_grad()

			containers, revealed = model(secrets, covers)

			original_wav = isdct_torch(covers.squeeze(0).squeeze(0), frame_length=4096, frame_step=62, window=torch.hamming_window)
			container_wav = isdct_torch(containers.squeeze(0).squeeze(0), frame_length=4096, frame_step=62, window=torch.hamming_window)
			container_2x = sdct_torch(container_wav, frame_length=4096, frame_step=62, window=torch.hamming_window).unsqueeze(0).unsqueeze(0)

			loss, loss_cover, loss_secret, loss_spectrum = StegoLoss(secrets, covers, containers, container_2x, revealed, beta)
			snr_audio = SNR(covers.cpu(), containers.cpu())
			psnr_image = PSNR(secrets, revealed)
			ssim_image = ssim(secrets, revealed)
			dtw_loss = softDTW(original_wav.cpu().unsqueeze(0), container_wav.cpu().unsqueeze(0))
			objective_loss = loss 
			if add_dtw_term: objective_loss += 10**(np.floor(np.log10(1/33791)) + 1) * dtw_loss
			with torch.autograd.set_detect_anomaly(True):
				objective_loss.backward()
			optimizer.step()

			train_loss.append(loss.detach().item())
			train_loss_cover.append(loss_cover.detach().item())
			train_loss_secret.append(loss_secret.detach().item())
			train_loss_spectrum.append(loss_spectrum.detach().item())
			snr.append(snr_audio)
			psnr.append(psnr_image.detach().item())
			ssim_secret.append(ssim_image.detach().item())
			train_dtw_loss.append(dtw_loss.detach().item())

			avg_train_loss = np.mean(train_loss[-slide:])
			avg_train_loss_cover = np.mean(train_loss_cover[-slide:])
			avg_train_loss_secret = np.mean(train_loss_secret[-slide:])
			avg_train_loss_spectrum = np.mean(train_loss_spectrum[-slide:])
			avg_snr = np.mean(snr[-slide:])
			avg_ssim = np.mean(ssim_secret[-slide:])
			avg_psnr = np.mean(psnr[-slide:])
			avg_dtw_loss = np.mean(train_dtw_loss[-slide:])

			print(
				f'(#{i})[{np.round(time.time()-ini,2)}s]\
				Train Loss {loss.detach().item()},\
				MSE audio {loss_cover.detach().item()},\
				MSE image {loss_secret.detach().item()},\
				MSE spectrum {loss_spectrum.detach().item()},\
				SNR {snr_audio},\
				PSNR {psnr_image.detach().item()},\
				SSIM {ssim_image.detach().item()},\
				DTW {dtw_loss.detach().item()}' 
			)

			# Log train average loss to wandb
			wandb.log({
				'tr_i_loss': avg_train_loss,
				'tr_i_cover_loss': avg_train_loss_cover,
				'tr_i_secret_loss': avg_train_loss_secret,
				'tr_i_spectrum_loss': avg_train_loss_spectrum,
				'SNR': avg_snr,
				'PSNR': avg_psnr,
				'SSIM': avg_ssim,
				'DTW': avg_dtw_loss,
			})

			# Log images
			if (i % 50 == 0) and (i != 0):
				avg_valid_loss, avg_valid_loss_cover, avg_valid_loss_secret, avg_valid_snr, avg_valid_psnr, avg_valid_ssim, avg_valid_dtw = validate(model, vd_loader, beta, dtw_criterion=softDTW, tr_i=i, epoch=epoch)
				
				vd_loss.append(avg_valid_loss) 
				vd_loss_cover.append(avg_valid_loss_cover) 
				vd_loss_secret.append(avg_valid_loss_secret) 
				vd_snr.append(avg_valid_snr) 
				vd_psnr.append(avg_valid_psnr)
				vd_ssim.append(avg_valid_ssim) 
				vd_dtw.append(avg_valid_dtw)

				is_best = bool(avg_train_loss < best_loss)
				# Save checkpoint if is a new best
				save_checkpoint({
					'epoch': epoch + 1,
					'state_dict': model.state_dict(),
					'best_loss': best_loss,
					'beta': beta,
					'lr': lr,
					'i': i + 1,
					'tr_loss': train_loss,
					'tr_cover_loss': train_loss_cover,
					'tr_loss_secret': train_loss_secret,
					'tr_snr': snr,
					'tr_psnr': psnr,
					'tr_ssim': ssim_secret,
					'tr_dtw': train_dtw_loss,
					'vd_loss': vd_loss,
					'vd_cover_loss': vd_loss_cover,
					'vd_loss_secret': vd_loss_secret,
					'vd_snr': vd_snr,
					'vd_psnr': vd_psnr,
					'vd_ssim': vd_ssim,
					'vd_dtw': vd_dtw,
				}, is_best=is_best, filename=os.path.join(os.environ.get('OUT_PATH'), f'models/checkpoint_run_{experiment}.pt'))
		
		print(
			f'Epoch [{epoch + 1}/{epochs}], \
			Average_loss: {avg_train_loss}, \
			Average_loss_cover: {avg_train_loss_cover}, \
			Average_loss_secret: {avg_train_loss_secret}, \
			Average_loss_spectrum: {avg_train_loss_spectrum}, \
			Average SNR: {avg_snr}, \
			Average PSNR: {avg_psnr},\
			Average SSIM: {avg_ssim}, \
			Average DTW: {avg_dtw_loss}'
		)

		# Log train average loss to wandb
		wandb.log({
			'tr_loss': avg_train_loss,
			'tr_cover_loss': avg_train_loss_cover,
			'tr_secret_loss': avg_train_loss_secret,
		})
		
		is_best = bool(avg_train_loss < best_loss)
		best_loss = min(avg_train_loss, best_loss)

		# Save checkpoint if is a new best
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'best_loss': best_loss,
			'beta': beta,
			'lr': lr,
			'i': i + 1,
		}, is_best=is_best, filename=os.path.join(os.environ.get('OUT_PATH'), f'models/checkpoint_run_{experiment}.pt'))

	print(f"Training took {time.time() - ini} seconds")
	torch.save(model.state_dict(), os.path.join(os.environ.get('OUT_PATH'), f'models/final_run_{experiment}.pt'))
	return model, avg_train_loss

def validate(model, vd_loader, beta, dtw_criterion=None, epoch=None, tr_i=None, verbose=False):

	device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'Using device: {device}')

	if torch.cuda.device_count() > 1:
  		print("Let's use", torch.cuda.device_count(), "GPUs!")
  		model = nn.DataParallel(model)

	model.to(device)

	# Set to evaluation mode
	model.eval()
	loss = 0

	valid_loss, valid_loss_cover, valid_loss_secret, valid_loss_spectrum, valid_snr, valid_psnr, valid_ssim, valid_dtw = [], [], [], [], [], [], [], []
	vd_datalen = len(vd_loader)
	
	iniv = time.time()
	with torch.no_grad():
		print('Validating current model...')
		for i, data in enumerate(vd_loader):

			secrets, covers = data[0].to(device), data[1].to(device)
			secrets = secrets.permute(0, 3, 1, 2).type(torch.cuda.FloatTensor)
			covers = covers.unsqueeze(1)

			containers, revealed = model(secrets, covers)

			if i == 0:
				fig = viz2paper(secrets.cpu(), revealed.cpu(), covers.cpu(), containers.cpu())
				wandb.log({f"Revelation at epoch {epoch}, vd iteration {tr_i}": fig})

			container_wav = isdct_torch(containers.squeeze(0).squeeze(0), frame_length=4096, frame_step=62, window=torch.hamming_window)
			container_2x = sdct_torch(container_wav, frame_length=4096, frame_step=62, window=torch.hamming_window).unsqueeze(0).unsqueeze(0)

			loss, loss_cover, loss_secret, loss_spectrum = StegoLoss(secrets, covers, containers, container_2x, revealed, beta)
			vd_snr_audio = SNR(covers.cpu(), containers.cpu())
			vd_psnr_image = PSNR(secrets, revealed)
			ssim_image = ssim(secrets, revealed)

			if dtw_criterion is not None:
				original_wav = isdct_torch(covers.squeeze(0).squeeze(0), frame_length=4096, frame_step=62, window=torch.hamming_window)
				dtw_loss = dtw_criterion(original_wav.cpu().unsqueeze(0), container_wav.cpu().unsqueeze(0))

			valid_loss.append(loss.detach().item())
			valid_loss_cover.append(loss_cover.detach().item())
			valid_loss_secret.append(loss_secret.detach().item())
			valid_loss_spectrum.append(loss_spectrum.detach().item())
			valid_snr.append(vd_snr_audio)
			valid_psnr.append(vd_psnr_image.detach().item())
			valid_ssim.append(ssim_image.detach().item())
			valid_dtw.append(dtw_loss.detach().item())

			print(
				f'(#{i})[{np.round(time.time()-iniv,2)}s]\
				Valid Loss {loss.detach().item()},\
				cover_error {loss_cover.detach().item()},\
				secret_error {loss_secret.detach().item()},\
				spectrum_error {loss_spectrum.detach().item()},\
				SNR {vd_snr_audio},\
				PSNR {vd_psnr_image.detach().item()},\
				SSIM {ssim_image.detach().item()},\
				DTW {dtw_loss.detach().item()}'
			)

			if i >= 2: break
			# if i >= vd_datalen: break

		avg_valid_loss = np.mean(valid_loss)
		avg_valid_loss_cover = np.mean(valid_loss_cover)
		avg_valid_loss_secret = np.mean(valid_loss_secret)
		avg_valid_loss_spectrum = np.mean(valid_loss_spectrum)
		avg_valid_snr = np.mean(valid_snr)
		avg_valid_psnr = np.mean(valid_psnr)
		avg_valid_ssim = np.mean(valid_ssim)
		avg_valid_dtw = np.mean(valid_dtw)

		wandb.log({
			'vd_loss': avg_valid_loss,
			'vd_cover_loss': avg_valid_loss_cover,
			'vd_secret_loss': avg_valid_loss_secret,
			'vd_spectrum_loss': avg_valid_loss_spectrum,
			'vd_SNR': avg_valid_snr,
			'vd_PSNR': avg_valid_psnr,
			'vd_SSIM': avg_valid_ssim,
			'vd_DTW': avg_valid_dtw
		})
		print(f"Validation took {time.time() - iniv} seconds")

	del valid_loss
	del valid_loss_cover
	del valid_loss_secret
	del valid_loss_spectrum
	del valid_snr
	del valid_psnr
	del valid_ssim
	del valid_dtw
	gc.collect()

	return avg_valid_loss, avg_valid_loss_cover, avg_valid_loss_secret, avg_valid_snr, avg_valid_psnr, avg_valid_ssim, avg_valid_dtw
			

if __name__ == '__main__':

	args = parser.parse_args()
	print(args)

	train_loader = loader(
		set='train', 
		rgb=args.rgb
	)
	test_loader = loader(
		set='test',
		rgb=args.rgb
	)

	model = StegoUNet(
		add_noise=args.add_noise, 
		noise_kind=args.noise_kind, 
		noise_amplitude=args.noise_amplitude
	)

	if args.from_checkpoint:
		# Load checkpoint
		checkpoint = torch.load(os.path.join(os.environ.get('OUT_PATH'),f'models/checkpoint_run_{args.experiment}.pt'), map_location='cpu')
		model = nn.DataParallel(model)
		model.load_state_dict(checkpoint['state_dict'])
		print('Checkpoint loaded ++')

	train(
		model=model, 
		tr_loader=train_loader, 
		vd_loader=test_loader, 
		beta=args.beta, 
		lr=args.lr, 
		epochs=8, 
		slide=15,
		prev_epoch=checkpoint['epoch'] if args.from_checkpoint else None,  
		prev_i=checkpoint['i'] if args.from_checkpoint else None,
		summary=args.summary,
		experiment=args.experiment,
		add_dtw_term=args.add_dtw_term
	)


'''

if __name__ == '__main__':
	args = parser.parse_args()
	train_loader = loader(set = 'train')
	test_loader = loader(set = 'test')
	device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")	
	print(f'Using device: {device}')
	chk = torch.load(os.path.join(os.environ.get('USER_PATH'), f'checkpoints/checkpoint_run_{int(args.experiment)}.pt'), map_location='cpu')
	model = StegoUNet()
	model = nn.DataParallel(model)
	model.load_state_dict(chk['state_dict'])
	model.to(device)
	print('Checkpoint loaded ++')
	#train(train_loader, beta = 0.3, lr = 0.001, epochs = 5, prev_epoch = chk['epoch'], prev_i = chk['i'])
	train(
		model=model, 
		tr_loader=train_loader, 
		vd_loader=test_loader, 
		beta=float(args.beta), 
		lr=float(args.lr), 
		epochs=15, 
		slide=15,
		prev_epoch=chk['epoch'], 
		prev_i=None,
		summary=args.summary,
		experiment=int(args.experiment)
	)

'''