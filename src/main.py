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
import matplotlib.pyplot as plt

MY_FOLDER = '/mnt/gpid07/imatge/margarita.geleta/pix2wav'
DATA_FOLDER = '/projects/deep_learning/ILSVRC/ILSVRC2012'
MY_DATA_FOLDER = f'{MY_FOLDER}/data'

print('PyTorch version:', torch.__version__)

class ImageProcessor():
    def __init__(self, image_path):
        self.image = Image.open(image_path).convert('RGB')

    def crop(self, proportion = 2 ** 6):
        nx, ny = self.image.size
        n = min(nx, ny)
        left = top = n / proportion
        right = bottom = (proportion - 1) * n / proportion
        self.image = self.image.crop((left, top, right, bottom))

    def scale(self, n = 250):
        self.image = self.image.resize((n, n), Image.ANTIALIAS)

    def forward(self):
        self.crop()
        self.scale()
        return self.image

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, folder: str, mappings: dict, extension: str = "JPEG"):
        self._data = pathlib.Path(root) / folder
        print(f'DATA LOCATED AT: {self._data}')
        self.extension = extension
        self._index = 0
        self._indices = []
        for key in mappings.keys():
            for img in glob.glob(f'{self._data}/{key}/*.{self.extension}'):
                self._indices.append(re.search(r'(?<=_)\d+', img).group())
                self._index += 1
        print('Set up done')

    def __len__(self):
        return self._index

    def __getitem__(self, index):
        img_path = glob.glob(f'{self._data}/{self._indices[index][0]}/*_{self._indices[index][1]}.{self.extension}')[0]
        print(f'READING: {img_path}')
        try:
            img = ImageProcessor(img_path).forward()
            return img
        except Exception as e:
            print(e)
            pass

def plot_samples(dataset):
    fig = plt.figure()
    for i in range(len(dataset)):
        sample = dataset[i]
        ax = plt.subplot(1, 6, i + 1)
        plt.tight_layout()
        ax.imshow(sample)
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        if i == 5:
            plt.show()
            break

if __name__ == '__main__':

    print('Preparing')
    mappings = {}
    with open(f'{MY_DATA_FOLDER}/mappings.txt') as f:
        for line in f:
            (key, i, img) = line.split()
            mappings[key] = img

    dataset = ImageDataset(DATA_FOLDER, 'train', mappings)

    plot_samples(dataset)

    with open(f'{MY_FOLDER}/data/imagenet_train.data', 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Dataset prepared')
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=64,
                                             shuffle=True,
                                             num_workers=4,
                                             pin_memory=True)
    print('Data loaded ++')

    # for i, batch in enumerate(dataloader):
    #    print(i, batch)

    print(len(dataloader.dataset))


