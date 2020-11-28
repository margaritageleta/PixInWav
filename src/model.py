import pathlib
import pickle
import torch
from PIL import Image
import glob as glob
import re
import itertools
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
#import matplotlib
import matplotlib.pyplot as plt

MY_FOLDER = '/mnt/gpid07/imatge/margarita.geleta/pix2wav'
DATA_FOLDER = '/projects/deep_learning/ILSVRC/ILSVRC2012'

if __name__ == '__main__':
    print('trying')
    with open(f'{MY_FOLDER}/data/imagenet_train.data', 'rb') as f:
        dataset = pickle.load(f)
    print(type(dataset))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=64,
                                             shuffle=True,
                                             num_workers=4,
                                             pin_memory=True)
    print('Done.')
    # for i, batch in enumerate(dataloader):
    #    print(i, batch)

    print(len(dataloader.dataset))
    print('HII')