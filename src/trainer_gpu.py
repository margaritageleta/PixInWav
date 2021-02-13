import os
import torch
import numpy as np
import torch.nn.functional as F
from loader import loader
from umodel import StegoUNet
from umodel_shuffle import StegoUNet_Shuffle
from modelsimple import Net
import torch.optim as optim
from time import time
import datetime
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from pystct import isdct

import gc # garbage collector

MY_FOLDER = '/mnt/gpid07/imatge/cristina.punti/PixInWav'
LOGDIR = os.path.join(f'{MY_FOLDER}/src/logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# LOGDIR = os.path.join("logs", '20201225-192204')
writer = SummaryWriter(log_dir=LOGDIR)
print(LOGDIR)

# assert(True == False)

def save_checkpoint(state, is_best, filename=f'{MY_FOLDER}/checkpoints/checkpoint.pt'):
     """Save checkpoint if a new best is achieved"""
     if is_best:
         print ("=> Saving a new best model")
         torch.save(state, filename)  # save checkpoint
     else:
         print ("=> Loss did not improve")

def compare_images(s, r, epoch):
	s = s.permute(0,2,3,1).detach().cpu().numpy().squeeze(0).astype(np.uint8)
	r = r.permute(0,2,3,1).detach().cpu().numpy().squeeze(0).astype(np.uint8)

	fig, ax = plt.subplots(1, 2, figsize=(10, 10))
	ax[0].imshow(s)
	ax[1].imshow(r)
	ax[0].set_title('Secret image')
	ax[1].set_title('Revealed image')
	ax[0].axis('off')
	ax[1].axis('off')
	plt.show()

	writer.add_figure('Image revelation', fig, epoch)

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

def train(model, dataloader, beta, lr, epochs=5, prev_epoch = None, prev_i = None):

	optimizer = optim.Adam(model.parameters(), lr=lr)

	ini = time()
	best_loss = np.inf
	datalen = len(dataloader)

	for epoch in range(epochs):

		if prev_epoch != None and epoch < prev_epoch - 1: continue

		train_loss, train_loss_cover, train_loss_secret = [], [], []
		model.train()

		for i, data in enumerate(dataloader):

			if prev_i != None and i < prev_i - 1: continue

			secrets, covers = data[0].to(device), data[1].to(device)

			secrets = secrets.unsqueeze(1).type(torch.cuda.FloatTensor)
			covers = covers.unsqueeze(1)

			optimizer.zero_grad()

			containers, revealed = model(secrets, covers)
			# revealed = model(covers)

			loss, loss_cover, loss_secret = stego_loss(secrets, covers, containers, revealed, beta)

			loss.backward()
			optimizer.step()

			train_loss.append(loss.detach().item())
			train_loss_cover.append(loss_cover.detach().item())
			train_loss_secret.append(loss_secret.detach().item())

			avg_train_loss = np.mean(train_loss)
			avg_train_loss_cover = np.mean(train_loss_cover)
			avg_train_loss_secret = np.mean(train_loss_secret)

			# Log train average loss to tensorboard
			writer.add_scalar(f'stego_{epoch + 1}/train_loss', avg_train_loss, i + 1)
			writer.add_scalar(f'cover_{epoch + 1}/train_loss', avg_train_loss_cover, i + 1)
			writer.add_scalar(f'secret_{epoch + 1}/train_loss', avg_train_loss_secret, i + 1)

			if i % 50 == 0:
				save_checkpoint({
					'epoch': epoch + 1,
					'state_dict': model.state_dict(),
					'best_loss': best_loss,
					'beta': beta,
					'lr': lr,
					'i': i + 1,
				}, is_best = True, filename=f'{MY_FOLDER}/checkpoints/checkpoint_test_gpu_unet_shuffle_run1_{epoch + 1}_{i + 1}.pt')

			print(('='* (i+1)) + f' {datalen - (i+1)} left to scan')
			print(f'Train Loss {loss.detach().item()}, cover_error {loss_cover.detach().item()}, secret_error {loss_secret.detach().item()}')
			print (f'Epoch [{epoch + 1}/{epochs}], Average_loss: {avg_train_loss}, Average_loss_cover: {avg_train_loss_cover}, Average_loss_secret: {avg_train_loss_secret}')

		# Log train average loss to tensorboard
		writer.add_scalar('stego/train_loss', avg_train_loss, epoch + 1)
		writer.add_scalar('cover/train_loss', avg_train_loss_cover, epoch + 1)
		writer.add_scalar('secret/train_loss', avg_train_loss_secret, epoch + 1)
		# Log images
		compare_images(secrets, revealed, epoch + 1)

		is_best = bool(avg_train_loss > best_loss)
		best_loss = min(avg_train_loss, best_loss)

		# Save checkpoint if is a new best
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'best_loss': best_loss,
			'beta': beta,
			'lr': lr,
		}, is_best)

	print(f"Training took {time() - ini} seconds")
	torch.save(model.state_dict(), f'{MY_FOLDER}/models/test_cris_1_{epochs}_run1.pt')
	return model, avg_train_loss


if __name__ == '__main__':

	device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'Using device: {device}')

	train_loader = loader(set = 'train')
	# test_loader = loader(set = 'test')

	# chk = torch.load(f'{MY_FOLDER}/checkpoints/checkpoint_run2_1_901.pt', map_location='cpu')
	model = StegoUNet()
	model.to(device)
	# model.load_state_dict(chk['state_dict'])

	# take one batch from the training loader
	secrets, covers = next(iter(train_loader))
	secrets = secrets.unsqueeze(1)
	covers = covers.unsqueeze(1)
	# print(secrets.shape, covers.shape)

	# We need to pass a batch of data along with the model 
	# Dibuix model
	#writer.add_graph(model, (secrets.detach(), covers.detach()), verbose = False)

	# train(train_loader, beta = 0.3, lr = 0.001, epochs = 5, prev_epoch = chk['epoch'], prev_i = chk['i'])
	train(model, train_loader, beta = 0.2, lr = 0.001, epochs = 5, prev_epoch = None, prev_i = None)