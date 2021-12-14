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
DATA_FOLDER = '/mnt/gpid08/datasets/ILSVRC/ILSVRC2012'
AUDIO_FOLDER ='/mnt/gpid07/imatge/cristina.punti/PixInWav/data/FSDnoisy18k.audio_'
#DATA_FOLDER = '/mnt/gpid08/datasets/coco-2017/coco/images'
#AUDIO_FOLDER_train = '/mnt/gpid08/users/teresa.domenech/train'
#AUDIO_FOLDER_val = '/mnt/gpid07/imatge/teresa.domenech/venv/PixInWav/PixInWav_dataset/val'
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
            )

    def forward(self, audio_path):
        self.sound, self.sr = torchaudio.load(audio_path)

        # Get the samples dimension
        sound = self.sound[0]
        # Create a temporary array
        tmp = torch.zeros([self._limit, ]).normal_(mean=0, std=0.005)
        # Cut the audio on limit
        if sound.numel() < self._limit:
            tmp[:sound.numel()] = sound[:]
        else:
            i = random.randint(0, len(sound) - self._limit)
            tmp[:] = sound[i:i + self._limit]
        if self._transform == 'cosine':
            return sdct_torch(
                tmp.type(torch.float32),
                frame_length=self._frame_length,
                frame_step=self._frame_step
            )
        elif self._transform == 'fourier':
            magnitude, phase = self.stft.transform(tmp.unsqueeze(0).type(torch.float32))
            return magnitude, phase

        else:
            raise Exception(f'Transform not implemented')


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
            mappings: dict,
            rgb: bool = True,
            transform: str = 'cosine',
            image_extension: str = "JPEG",
            audio_extension: str = "wav"
    ):

        # self._image_data_path = pathlib.Path(image_root) / folder
        self._image_data_path = pathlib.Path(image_root) / 'train'
        self._audio_data_path = pathlib.Path(f'{audio_root}{folder}')
        self._MAX_LIMIT = 10000 if folder == 'train' else 900
        self._MAX_AUDIO_LIMIT = 17584 if folder == 'train' else 946
        self._colorspace = 'RGB' if rgb else 'L'
        self._transform = transform

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

        self._AUDIO_PROCESSOR = AudioProcessor(transform=self._transform)

        print('Set up done')

    def __len__(self):
        return self._index

    def __getitem__(self, index):
        key = self._indices[index][0]
        indexer = self._indices[index][1]
        rand_indexer = random.randint(0, self._MAX_AUDIO_LIMIT - 1)

        img_path = glob.glob(f'{self._image_data_path}/{key}/{key}_{indexer}.{self.image_extension}')[0]
        audio_path = self._audios[rand_indexer]

        img = np.asarray(ImageProcessor(image_path=img_path, colorspace=self._colorspace).forward()).astype('float64')

        if self._transform == 'cosine':
            sound_stct = self._AUDIO_PROCESSOR.forward(audio_path)
            return (img, sound_stct)
        elif self._transform == 'fourier':
            magnitude_stft, phase_stft = self._AUDIO_PROCESSOR.forward(audio_path)
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
    mappings = {}
    with open(f'{MY_DATA_FOLDER}/mappings.txt') as f:
        for line in f:
            (key, i, img) = line.split()
            mappings[key] = img

    dataset = StegoDataset(
        image_root=DATA_FOLDER,
        audio_root=AUDIO_FOLDER,
        folder=set,
        mappings=mappings,
        rgb=rgb,
        transform=transform
    )

    print('Dataset prepared.')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print('Data loaded ++')
    return dataloader