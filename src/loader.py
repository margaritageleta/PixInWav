import re
import torch
import random
import pathlib
import glob as glob
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

MY_FOLDER = '/mnt/gpid07/imatge/margarita.geleta/pix2wav'
DATA_FOLDER = '/projects/deep_learning/ILSVRC/ILSVRC2012'
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

    def forward(self):
        self.crop()
        self.scale()
        return self.image

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_root: str,
                 audio_root: str,
                 folder: str,
                 mappings: dict,
                 image_extension: str = "JPEG",
                 audio_extension: str = "wav"):

        self._image_data_path = pathlib.Path(image_root) / folder
        self._audio_data_path = pathlib.Path(f'{audio_root}{folder}')
        self._MAX_LIMIT = 1000

        print(f'IMAGE DATA LOCATED AT: {self._image_data_path}')
        print(f'AUDIO DATA LOCATED AT: {self._audio_data_path}')

        self.image_extension = image_extension
        self.audio_extension = audio_extension
        self._index = 0
        self._indices = []
        self._audios = []

        for key in mappings.keys():
            for img in glob.glob(f'{self._image_data_path}/{key}/*.{self.image_extension}'):
                self._indices.append(re.search(r'(?<=_)\d+', img).group())
                self._index += 1
                if self._index == self._MAX_LIMIT:
                    break
            if self._index == self._MAX_LIMIT:
                break

        _aux_index = 0
        for audio_path in glob.glob(f'{self._audio_data_path}/*.{self.audio_extension}'):
            self._audios.append(audio_path)
            _aux_index += 1
            if _aux_index == self._index: break

        print('Set up done')

    def __len__(self):
        return self._index

    def __getitem__(self, index):
        img_path = glob.glob(f'{self._image_data_path}/{self._indices[index][0]}/*_{self._indices[index][1]}.{self.image_extension}')[0]
        # try:
        img = ImageProcessor(img_path).forward()

        return img
        # except Exception as e:
        # print(e)
        # pass


if __name__ == '__main__':

    print('Preparing')
    mappings = {}
    with open(f'{MY_DATA_FOLDER}/mappings.txt') as f:
        for line in f:
            (key, i, img) = line.split()
            mappings[key] = img

    dataset = ImageDataset(image_root = DATA_FOLDER,
                           audio_root = AUDIO_FOLDER,
                           folder ='train',
                           mappings = mappings)

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
