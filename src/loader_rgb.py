import os
import re
import torch
import random
import pathlib
import torchaudio
import numpy as np
import glob as glob
from PIL import Image
from torch.utils.data import DataLoader

from pystct import sdct_torch, isdct_torch
import matplotlib.pyplot as plt

MY_FOLDER = os.environ.get('USER_PATH')
DATA_FOLDER = os.environ.get('IMAGE_PATH')
AUDIO_FOLDER = f'{MY_FOLDER}/data/FSDnoisy18k.audio_'
MY_DATA_FOLDER = f'{MY_FOLDER}/data'

class ImageProcessor():
    def __init__(self, image_path):
        self.image = Image.open(image_path).convert('RGB')

    def crop(self, proportion = 2 ** 6):
        nx, ny = self.image.size
        n = min(nx, ny)
        left = top = n / proportion
        right = bottom = (proportion - 1) * n / proportion
        self.image = self.image.crop((left, top, right, bottom))

    def scale(self, n = 256):
        self.image = self.image.resize((n, n), Image.ANTIALIAS)

    def normalize(self):
        self.image = np.array(self.image).astype('float') / 255.0

    def forward(self):
        self.crop()
        self.scale()
        self.normalize()
        return self.image

class AudioProcessor():
    def __init__(self, audio_path):
        self.sound, self.sr = torchaudio.load(audio_path)
        # Corresponds to 1.5 seconds approximately
        self._limit = 67522 # 2 ** 16 + 2 ** 11 - 1
        self._frame_length = 2 ** 12
        self._frame_step = 2 ** 6 - 2

    def forward(self):
        # Get the samples dimension
        sound = self.sound[0]
        # Create a temporary array
        tmp = torch.zeros([self._limit, ]).normal_(mean = 0, std = 0.005)
        # Cut the audio on limit
        if sound.numel() < self._limit:
            tmp[:sound.numel()] = sound[:]
        else:
            i = random.randint(0, len(sound) - self._limit)
            tmp[:] = sound[i:i + self._limit]
        sound_stct = sdct_torch(tmp.type(torch.float32),
                          frame_length = self._frame_length,
                          frame_step = self._frame_step)
        return sound_stct

class StegoDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_root: str,
                 audio_root: str,
                 folder: str,
                 mappings: dict,
                 image_extension: str = "JPEG",
                 audio_extension: str = "wav"):

        # self._image_data_path = pathlib.Path(image_root) / folder
        self._image_data_path = pathlib.Path(image_root) / 'train'
        self._audio_data_path = pathlib.Path(f'{audio_root}{folder}')
        self._MAX_LIMIT = 10000 if folder == 'train' else 900
        self._MAX_AUDIO_LIMIT = 17584 if folder == 'train' else 946

        print(f'IMAGE DATA LOCATED AT: {self._image_data_path}')
        print(f'AUDIO DATA LOCATED AT: {self._audio_data_path}')

        self.image_extension = image_extension
        self.audio_extension = audio_extension
        self._index = 0
        self._indices = []
        self._audios = []

        test_i = 0
        for key in mappings.keys():
            for img in glob.glob(f'{self._image_data_path}/{key}/*.{self.image_extension}'):
                test_i += 1
                if folder == 'train' or (folder == 'test' and test_i > self._MAX_LIMIT):
                    self._indices.append((key, re.search(r'(?<=_)\d+', img).group()))
                    self._index += 1
                if self._index == self._MAX_LIMIT:
                    break
            if self._index == self._MAX_LIMIT:
                break

        _aux_index = 0
        for audio_path in glob.glob(f'{self._audio_data_path}/*.{self.audio_extension}'):
            self._audios.append(audio_path)
            _aux_index += 1
            if _aux_index == self._MAX_AUDIO_LIMIT: break
        random.shuffle(self._audios)

        print('Set up done')

    def __len__(self):
        return self._index

    def __getitem__(self, index):
        key = self._indices[index][0]
        indexer = self._indices[index][1]
        rand_indexer = random.randint(0, self._MAX_AUDIO_LIMIT - 1)

        img_path = glob.glob(f'{self._image_data_path}/{key}/{key}_{indexer}.{self.image_extension}')[0]
        # print(f'{rand_indexer} < {self._MAX_AUDIO_LIMIT}')
        audio_path = self._audios[rand_indexer]

        img = np.asarray(ImageProcessor(img_path).forward()).astype('float64')
        sound_stct = AudioProcessor(audio_path).forward()

        return (img, sound_stct)

def loader(set = 'train'):
    print('Preparing')
    mappings = {}
    with open(f'{MY_DATA_FOLDER}/mappings.txt') as f:
        for line in f:
            (key, i, img) = line.split()
            mappings[key] = img

    dataset = StegoDataset(image_root=DATA_FOLDER,
                           audio_root=AUDIO_FOLDER,
                           folder=set,
                           mappings=mappings)
    print('Dataset prepared')
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=True,
                                             num_workers=4,
                                             pin_memory=True)
    print('Data loaded ++')
    return dataloader

if __name__ == '__main__':

    #train_loader = loader(set = 'train')
    test_loader = loader(set = 'test')

    #for i, batch in enumerate(train_loader):
    #    print(f'Batch {i}, shape image {batch[0].shape}, shape audio {batch[1].shape}')
    #    if i == 1: break
    print(len(test_loader))
    for i, batch in enumerate(test_loader):
        print(f'Batch {i}, shape image {batch[0].shape}, shape audio {batch[1].shape}')
        # if i == 1: break

    print(len(train_loader.dataset))
    print(len(test_loader.dataset))
