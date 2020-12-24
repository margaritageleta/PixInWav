import torch
import torch.nn.functional as F


def stego_loss(secret, cover, container, revealed, beta):
	loss_cover = F.mse_loss(cover, container)
	loss_secret = F.mse_loss(secret, revealed)
	loss = loss_cover + beta * loss_secret
	return loss, loss_cover, loss_secret

if __name__ == "__main__":
	print('This is the trainer.py')