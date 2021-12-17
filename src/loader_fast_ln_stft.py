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
from torch_stft import STFT
import matplotlib.pyplot as plt

MY_FOLDER = '/mnt/gpid07/imatge/teresa.domenech/venv/PixInWav'
DATA_FOLDER = '/mnt/gpid08/datasets/coco-2017/coco/images'
AUDIO_FOLDER ='/mnt/gpid08/users/teresa.domenech'
MY_DATA_FOLDER = f'{MY_FOLDER}/notebooks'
fiAudio = False


class ImageProcessor():
    """
    Function to preprocess the images from the custom
    dataset. It includes a series of transformations:

    - At __init__ we convert the image to the desired [colorspace].
    - Crop function crops the image to the desired [proportion].
    - Scale scales the images to desired size [n]x[n].
    - Normalize performs the normalization of the channels.
    """

    def __init__(self, image_path, colorspace='RGB'):
        self.image = Image.open(image_path).convert(colorspace)

    def crop(self, proportion=2 ** 6):
        nx, ny = self.image.size
        n = min(nx, ny)
        left = top = n / proportion
        right = bottom = (proportion - 1) * n / proportion
        self.image = self.image.crop((left, top, right, bottom))

    def scale(self, n=256):
        self.image = self.image.resize((n, n), Image.ANTIALIAS)

    def normalize(self):
        self.image = np.array(self.image).astype('float') / 255.0

    def forward(self):
        self.crop()
        self.scale()
        self.normalize()
        return self.image


class AudioProcessor():
    """
    Function to preprocess the audios from the custom
    dataset. We set the [_limit] in terms of samples,
    the [_frame_length] and [_frame_step] of the [transform]
    transform.
    If transform is [cosine] it returns just the STDCT matrix.
    Else, if transform is [fourier] returns the STFT magnitude
    and phase.
    """

    def __init__(self, transform):
        # Corresponds to 1.5 seconds approximately
        self._limit = 67522  # 2 ** 16 + 2 ** 11 - 2 ** 6 + 2
        self._frame_length = 2 ** 12 if transform == 'cosine' else 2 ** 11 - 1
        self._frame_step = 2 ** 6 - 2 if transform == 'cosine' else 132

        self._transform = transform
        if self._transform == 'fourier':
            self.stft = STFT(
                filter_length=self._frame_length,
                hop_length=self._frame_step,
                win_length=self._frame_length,
                window='hann'
            ).to(device)

    def forward(self, audio_path):

        #self.sound, self.sr = torchaudio.load(torch.from_numpy(audio_path))
        return None

class StegoDataset(torch.utils.data.Dataset):
    """
    Custom datasets pairing images with spectrograms.
    - [image_root] defines the path to read the images from.
    - [audio_root] defines the path to read the audio clips from.
    - [folder] can be either [train] or [test].
    - [mappings] is the dictionary containing a descriptive name for
    the images from ImageNet. It is used to index the different
    subdirectories.
    - [rgb] is a boolean that indicated whether we are using color (RGB)
    images or black and white ones (B&W).
    - [transform] defines the transform to use to process audios. Can be
    either [cosine] or [fourier].
    - [image_extension] defines the extension of the image files.
    By default it is set to JPEG.
    - [audio_extension] defines the extension of the audio files.
    By default it is set to WAV.
    """

    def __init__(
            self,
            image_root: str,
            audio_root: str,
            folder: str,
            #mappings: dict,
            rgb: bool = True,
            transform: str = 'cosine',
            image_extension: str = "jpg",
            audio_extension: str = "wav"
    ):
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._image_data_path = pathlib.Path(image_root) / folder
        if folder == 'train2017':
            self._audio_data_path = pathlib.Path(audio_root) / 'train'
        else:
            self._audio_data_path = pathlib.Path(audio_root) / 'val'

        self._MAX_LIMIT = 10000 if folder == 'train2017' else 900 #10000 #900
        self._MAX_AUDIO_LIMIT = 5224 if folder == 'train2017' else 900 #5224 #946
        self._colorspace = 'RGB' if rgb else 'L'
        self._transform = transform

        print(f'IMAGE DATA LOCATED AT: {self._image_data_path}')
        print(f'AUDIO DATA LOCATED AT: {self._audio_data_path}')

        self.image_extension = image_extension
        self.audio_extension = audio_extension
        self._audios = []
        self._audios_p = []
        self._images = []

        _aux_index_i = 0
        for image_path in glob.glob(f'{self._image_data_path}/*.{self.image_extension}'):
            self._images.append(image_path)
            _aux_index_i += 1
            if _aux_index_i == self._MAX_LIMIT: break
        self._images = self._images
        random.shuffle(self._images)


        print('entra al else')
        #self._audios = np.load('/mnt/gpid08/users/teresa.domenech/audio_cosine.npy').to(device)
        if folder == 'train2017':
            self._audios= np.load('/mnt/gpid08/users/teresa.domenech/audio_fourier_train.npy')
            self._audios_p = np.load('/mnt/gpid08/users/teresa.domenech/audio_fourier_train_phase.npy')
        else:
            self._audios = np.load('/mnt/gpid08/users/teresa.domenech/audio_fourier_val.npy')
            self._audios_p = np.load('/mnt/gpid08/users/teresa.domenech/audio_fourier_val_phase.npy')
        print('load fet')
        #random.shuffle(self._audios)
        #self._AUDIO_PROCESSOR = AudioProcessor(transform=self._transform)

        print('Set up done')

    def __len__(self):
        return self._MAX_LIMIT

    def __getitem__(self,index):
        rand_indexer = random.randint(0, self._MAX_AUDIO_LIMIT - 1)
        #rand_indexer_i = random.randint(0, self._MAX_LIMIT - 1)

        # per la imatge s'hauria de fer servir index
        img_path = self._images[index]
        #audio_path = self._audios[rand_indexer]
        audio_path_m = self._audios[rand_indexer]
        audio_path_p = self._audios_p[rand_indexer]

        img = np.asarray(ImageProcessor(image_path=img_path, colorspace=self._colorspace).forward()).astype('float64')

        if self._transform == 'cosine':
            print('entra aqui')
            sound_stct = torch.from_numpy(audio_path)
            print('llegit')
            return (img, sound_stct)
        elif self._transform == 'fourier':
            magnitude_stft = torch.from_numpy(audio_path_m)
            phase_stft = torch.from_numpy(audio_path_p)
            #magnitude_stft, phase_stft = self._AUDIO_PROCESSOR.forward(audio_path)
            return (img, magnitude_stft, phase_stft)
        else:
            raise Exception(f'Transform not implemented')


def loader(set='train', rgb=True, transform='cosine'):
    """
    Prepares the custom dataloader.
    - [set] defines the set type. Can be either [train] or [test].
    - [rgb] is a boolean that indicated whether we are using color (RGB)
    images or black and white ones (B&W).
    - [transform] defines the transform to use to process audios. Can be
    either [cosine] or [fourier].
    """
    print('Preparing dataset...')
    '''
    mappings = {}
    with open(f'{MY_DATA_FOLDER}/untitled.txt') as f:
        for line in f:
            (key, i, img) = line.split()
            mappings[key] = img
    '''
    dataset = StegoDataset(
        image_root=DATA_FOLDER,
        audio_root=AUDIO_FOLDER,
        folder=set,
        #mappings=mappings,
        rgb=rgb,
        transform=transform
    )
    print(len(dataset), 'len datasets')
    print('Dataset prepared.')
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )#.to(device)

    print('Data loaded ++')
    return dataloader
