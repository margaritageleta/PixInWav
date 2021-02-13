import os
import gc
import torch
import wandb
import numpy as np
from time import time
from loader import loader
import torch.optim as optim
from umodel import StegoUNet
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pystct import isdct

# assert(True == False)

def save_checkpoint(state, is_best, filename=os.path.join(os.environ.get('USER_PATH'),'/data/checkpoints/checkpoint.pt')):
	 """Save checkpoint if a new best is achieved"""
	 if is_best:
		 print ("=> Saving a new best model")
		 torch.save(state, filename)  # save checkpoint
	 else:
		 print ("=> Loss did not improve")

def compare_images(s, r):
	s = s.permute(0,2,3,1).detach().numpy().squeeze(0).astype(np.uint8)
	r = r.permute(0,2,3,1).detach().numpy().squeeze(0).astype(np.uint8)

	fig, ax = plt.subplots(1, 2, figsize=(10, 10))
	ax[0].imshow(s)
	ax[1].imshow(r)
	ax[0].set_title('Secret image')
	ax[1].set_title('Revealed image')
	ax[0].axis('off')
	ax[1].axis('off')
	plt.show()

	return fig

def stego_loss(secret, cover, container, revealed, beta):

	loss_cover = F.mse_loss(cover, container)
	loss_secret = F.mse_loss(secret, revealed)
	loss = (1 - beta) * loss_cover + beta * loss_secret
	return loss, loss_cover, loss_secret

def stego_loss_wav(secret, cover, container, revealed, beta):
	cover_wav = torch.tensor(isdct(cover.squeeze(0).squeeze(0).detach().numpy(), frame_step=62))
	container_wav = torch.tensor(isdct(container.squeeze(0).squeeze(0).detach().numpy(), frame_step=62))
	loss_cover = torch.autograd.Variable(torch.abs(cover_wav - container_wav).sum(), requires_grad=True)
	loss_secret = F.mse_loss(secret, revealed)

	loss = (1 - beta) * loss_cover + beta * loss_secret
	return loss, loss_cover, loss_secret

def train(model, tr_loader, vd_loader, beta, lr, epochs=5, prev_epoch = None, prev_i = None):

	wandb.init(project='PixInWav')
	wandb.watch(model)

	device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'Using device: {device}')
	model.to(device)

	# Set to training mode
	model.train()

	# This is the number of parameters used in the model
	num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f'Number of model parameters: {num_params}')

	optimizer = optim.Adam(model.parameters(), lr=lr)

	ini = time()
	best_loss = np.inf
	datalen = len(tr_loader)

	for epoch in range(epochs):

		if prev_epoch != None and epoch < prev_epoch - 1: continue

		train_loss, train_loss_cover, train_loss_secret = [], [], []

		for i, data in enumerate(tr_loader):

			if prev_i != None and i < prev_i - 1: continue

			secrets, covers = data[0].to(device), data[1].to(device)
			secrets = secrets.unsqueeze(1).type(torch.cuda.FloatTensor)
			covers = covers.unsqueeze(1)

			optimizer.zero_grad()

			containers, revealed = model(secrets, covers)

			loss, loss_cover, loss_secret = stego_loss(secrets, covers, containers, revealed, beta)

			loss.backward()
			optimizer.step()

			train_loss.append(loss.detach().item())
			train_loss_cover.append(loss_cover.detach().item())
			train_loss_secret.append(loss_secret.detach().item())

			avg_train_loss = np.mean(train_loss)
			avg_train_loss_cover = np.mean(train_loss_cover)
			avg_train_loss_secret = np.mean(train_loss_secret)

			print(f'Train Loss {loss.detach().item()}, cover_error {loss_cover.detach().item()}, secret_error {loss_secret.detach().item()}')

			# Log train average loss to wandb
			wandb.log({
				'tr_loss': avg_train_loss,
				'tr_cover_loss': avg_train_loss_cover,
				'tr_secret_div': avg_train_loss_secret,
			})

			# Log images
			if i % 50 == 0:
				fig = compare_images(secrets, revealed)
				wandb.log({f"Image revelation at epoch {epoch}": fig})
				validate(model, vd_loader)
		
		print (f'Epoch [{epoch + 1}/{epochs}], Average_loss: {avg_train_loss}, Average_loss_cover: {avg_train_loss_cover}, Average_loss_secret: {avg_train_loss_secret}')

		is_best = bool(avg_train_loss > best_loss)
		best_loss = min(avg_train_loss, best_loss)

		# Save checkpoint if is a new best
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'best_loss': best_loss,
			'beta': beta,
			'lr': lr,
		}, is_best=is_best, filename=os.path.join(os.environ.get('USER_PATH'), 'checkpoints/checkpoint.pt'))

	print(f"Training took {time() - ini} seconds")
	torch.save(model.state_dict(), os.path.join(os.environ.get('USER_PATH'), 'checkpoints/final.pt'))
	return model, avg_train_loss

def validate(model, vd_loader, epoch=None, verbose=False):

	device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'Using device: {device}')

	model.to(device)

	# Set to evaluation mode
	model.eval()
	loss = 0

	total_vae_loss, total_rec_loss, total_KL_div  = [], [], []
	total_L1_loss, total_zeros_loss, total_ones_loss = [], [], []
	
	ini = time.time()
	with torch.no_grad():
		print('Validating current model...')
		for i, data in enumerate(vd_loader):

			secrets, covers = data[0].to(device), data[1].to(device)
			secrets = secrets.unsqueeze(1).type(torch.cuda.FloatTensor)
			covers = covers.unsqueeze(1)

			containers, revealed = model(secrets, covers)

			loss, loss_cover, loss_secret = stego_loss(secrets, covers, containers, revealed, beta)

			valid_loss.append(loss.detach().item())
			valid_loss_cover.append(loss_cover.detach().item())
			valid_loss_secret.append(loss_secret.detach().item())

			avg_valid_loss = np.mean(valid_loss)
			avg_valid_loss_cover = np.mean(valid_loss_cover)
			avg_valid_loss_secret = np.mean(valid_loss_secret)

			wandb.log({
				'vd_loss': avg_valid_loss,
				'vd_cover_loss': avg_valid_loss_cover,
				'vd_secret_div': avg_valid_loss_secret,
			})
			print(f'Valid Loss {loss.detach().item()}, cover_error {loss_cover.detach().item()}, secret_error {loss_secret.detach().item()}')
			

if __name__ == '__main__':

	train_loader = loader(set = 'train')
	test_loader = loader(set = 'test')

	# chk = torch.load(f'{MY_FOLDER}/checkpoints/checkpoint_run2_1_901.pt', map_location='cpu')
	model = StegoUNet()
	# model.load_state_dict(chk['state_dict'])

	# train(train_loader, beta = 0.3, lr = 0.001, epochs = 5, prev_epoch = chk['epoch'], prev_i = chk['i'])
	train(
		model=model, 
		tr_loader=train_loader, 
		vd_loader=test_loader, 
		beta = 0.2, lr = 0.001, epochs = 5, prev_epoch = None, prev_i = None)