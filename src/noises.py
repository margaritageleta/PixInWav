import torch
import numpy as np

def gaussian_noise(audio, mean=0.0, std=1.0):
    """
    Generates Gaussian noise of audio length
    with mean and std defined in the params.
    """
    return mean + torch.from_numpy(np.random.randn(audio.size()[0])) * std

def salt_noise(audio, prob=0.005):
    """
    Generates Salt noise of audio length
    with probability defined in the params.
    """
    ymax, ymin = audio.max(), audio.min()
    random = torch.from_numpy(np.random.uniform(size=audio.size()[0]))
    audio[(random < prob)] = ymax
    return audio 

def pepper_noise(audio, prob=0.005):
    """
    Generates Pepper noise of audio length
    with probability defined in the params.
    """
    ymax, ymin = audio.max(), audio.min()
    random = torch.from_numpy(np.random.uniform(size=audio.size()[0]))
    audio[(random > 1 - prob)] = ymin
    return audio

def salt_and_pepper_noise(audio, prob=0.005):
    """
    Generates Salt & Pepper noise of audio length
    with probability defined in the params.
    """
    ymax, ymin = audio.max(), audio.min()
    random = torch.from_numpy(np.random.uniform(size=audio.size()[0]))
    audio[(random < prob)] = ymax
    audio[(random > 1 - prob)] = ymin
    return audio

def add_noise(audio, noise_kind, noise_amplitude=0.01):
    """
    Adds [noise_kind] noise to waveform audio
    with amplitude equal to [noise_amplitude].

    Available noise kinds:
    - Gaussian      [gaussian]
    - Speckle       [speckle]
    - Salt          [salt]
    - Pepper        [pepper]
    - Salt & Pepper [salt&pepper]

    N.B. for Salt, Pepper and Salt&Pepper the noise
    amplitude is converted to a probability with
    tanh activation function.
    """
    if   noise_kind == 'gaussian': return audio + noise_amplitude * gaussian_noise(audio)    
    elif noise_kind == 'speckle': return audio * (1 + noise_amplitude * gaussian_noise(audio))
    elif noise_kind == 'salt': return salt_noise(audio, np.tanh(noise_amplitude))
    elif noise_kind == 'pepper': return pepper_noise(audio, np.tanh(noise_amplitude))
    elif noise_kind == 'salt&pepper': return salt_and_pepper_noise(audio, np.tanh(noise_amplitude))
    else: raise Exception(f'Noise not implemented')