import torch
import numpy as np
import torch.nn.functional as F
from loader import loader
from model import StegoNet
import torch.optim as optim
from time import time

import gc # garbage collector

def stego_loss(secret, cover, container, revealed, beta):

	loss_cover = F.mse_loss(cover, container)
	loss_secret = F.mse_loss(secret, revealed)
	loss = loss_cover + beta * loss_secret
	return loss, loss_cover, loss_secret

def train(dataloader, beta, lr, epochs=5):
	optimizer = optim.Adam(model.parameters(), lr=lr)

	ini = time()
	for epoch in range(epochs):

		train_loss = []
		model.train()

		for i, data in enumerate(dataloader):
			secrets, covers = data[0], data[1]
			secrets = secrets.permute(0,3,1,2).type(torch.FloatTensor)
			covers = covers.unsqueeze(1)

			optimizer.zero_grad()
			containers, revealed = model(secrets, covers)

			loss, loss_cover, loss_secret = stego_loss(secrets, covers, containers, revealed, beta)

			loss.backward()
			optimizer.step()

			train_loss.append(loss.detach().item())

			avg_train_loss = np.mean(train_loss)

			print(f'Train Loss {loss.detach().item()}, cover_error {loss_cover.detach().item()}, secret_error {loss_secret.detach().item()}')
			print (f'Epoch [{epoch + 1}/{epochs}], Average_loss: {avg_train_loss}')

	print(f"Training took {time() - ini} seconds")
	return model, avg_train_loss


if __name__ == '__main__':

	train_loader = loader(set = 'train')
	# test_loader = loader(set = 'test')

	model = StegoNet()

	train(train_loader, beta = 0.3, lr = 0.001, epochs = 5)