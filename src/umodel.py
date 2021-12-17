import torch
import numpy as np
import torch.nn as nn
from torch import utils
import torch.nn.functional as F
from pystct import sdct_torch, isdct_torch
from noises import add_noise
import random
from torch_stft import STFT

def fix_permutation_idx(input):
    idx = torch.randperm(input.nelement())
    return idx

def fix_permutation(input,idx):
    input = input.reshape(-1)[idx].view(input.size())
    return input


def permutation(input):
    idx = torch.randperm(input.nelement())
    input = input.reshape(-1)[idx].view(input.size())
    return input, idx

def unpermutation(input,idx):
    idx_inv = torch.argsort(idx)
    output = input.reshape(-1)[idx_inv].view(input.size())
    return output

def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''
        return pixel_unshuffle(input, self.downscale_factor)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.8, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.8, inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Down(nn.Module):

    def __init__(self, in_channels, out_channels, downsample_factor=8, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels
        self.conv = DoubleConv(in_channels, out_channels, mid_channels)
        self.down = nn.MaxPool2d(downsample_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.down(x)
        return x

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None, image_alone = False):
        super().__init__()
        self.image_alone = image_alone
        if not mid_channels:
            mid_channels = out_channels
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels , mid_channels, kernel_size=3, stride=4, output_padding=0),
            nn.LeakyReLU(0.8, inplace=True),
            nn.ConvTranspose2d(mid_channels , out_channels, kernel_size=3, stride=2, output_padding=1),
            nn.LeakyReLU(0.8, inplace=True),
        )
        self.conv = DoubleConv(out_channels * 2 if self.image_alone else out_channels * 3, out_channels, mid_channels)

    def forward(self, mix, im_opposite, au_opposite = None):
        mix = self.up(mix)
        x = torch.cat((mix, im_opposite), dim=1) if self.image_alone else torch.cat((au_opposite, mix, im_opposite), dim=1)
        return self.conv(x)

class ScalarMultiply(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.weight, 0.1, 1.0)

    def forward(self, x):
        return torch.mul(x, self.weight)

class PrepHidingNet(nn.Module):
    def __init__(self, architecture='resindep', transform='cosine'):
        super(PrepHidingNet, self).__init__()
        self._architecture = architecture
        self._transform = transform
        
        self.pixel_shuffle = nn.PixelShuffle(2)
        if self._architecture == 'resscale':
            self.scale = ScalarMultiply()

        elif (self._architecture == 'resindep') or (self._architecture == 'resdep') or (self._architecture == 'plaindep'):
            self.im_encoder_layers = nn.ModuleList([
                Down(1, 64),
                Down(64, 64 * 2)
            ])
            self.im_decoder_layers = nn.ModuleList([
                Up(64 * 2, 64, image_alone=True),
                Up(64, 1, image_alone=True)
            ])
            if (self._architecture == 'resdep') or (self._architecture == 'plaindep'):
             self.au_encoder_layers = nn.ModuleList([
                Down(1, 64),
                Down(64, 64 * 2)
            ])
        else: raise Exception('Unknown architecture')
        

    def forward(self, im, au=None):

        im = self.pixel_shuffle(im)

        if self._transform == 'cosine':
            im_enc = [nn.Upsample(scale_factor=(8, 2), mode='bilinear', align_corners=True)(im)]
        elif self._transform == 'fourier':
            im_enc = [nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True)(im)]
        else: raise Exception(f'Transform not implemented')

        if self._architecture == 'resscale':
            return self.scale(im_enc[0])

        elif self._architecture in ['resindep', 'resdep', 'plaindep']:

            if (self._architecture == 'resdep') or (self._architecture == 'plaindep'):
                au_enc = [au]
                for enc_layer_idx, enc_layer in enumerate(self.au_encoder_layers):
                    au_enc.append(enc_layer(au_enc[-1]))
            
            for enc_layer_idx, enc_layer in enumerate(self.im_encoder_layers):
                im_enc.append(enc_layer(im_enc[-1]))

            mix_dec = [im_enc.pop(-1)] if (self._architecture == 'resindep') else [au_enc.pop(-1) + im_enc.pop(-1)]

            for dec_layer_idx, dec_layer in enumerate(self.im_decoder_layers):
                mix_dec.append(
                    dec_layer(
                        mix_dec[-1], 
                        im_enc[-1 - dec_layer_idx],
                        None if (self._architecture == 'resindep') else au_enc[-1 - dec_layer_idx]
                    )
                )
            return mix_dec[-1]
        

class RevealNet(nn.Module):
    def __init__(self):
        super(RevealNet, self).__init__()

        self.pixel_unshuffle = PixelUnshuffle(2)

        self.im_encoder_layers = nn.ModuleList([
            Down(1, 64),
            Down(64, 64 * 2)
        ])

        self.im_decoder_layers = nn.ModuleList([
            Up(64 * 2, 64, image_alone=True),
            Up(64, 1, image_alone=True)
        ])

    def forward(self, ct):
        im_enc = [F.interpolate(ct, size=(256 * 2, 256 * 2))]

        for enc_layer_idx, enc_layer in enumerate(self.im_encoder_layers):
            im_enc.append(enc_layer(im_enc[-1]))
        
        im_dec = [im_enc.pop(-1)]

        for dec_layer_idx, dec_layer in enumerate(self.im_decoder_layers):
            im_dec.append(
                dec_layer(im_dec[-1], 
                im_enc[-1 - dec_layer_idx])
            )
        
        # Pixel Unshuffle and delete 4th component
        revealed = torch.narrow(self.pixel_unshuffle(im_dec[-1]), 1, 0, 3)

        return revealed


class StegoUNet(nn.Module):
    def __init__(
        self, 
        transform='cosine',
        add_noise=False, 
        noise_kind=None, 
        noise_amplitude=None, 
        frame_length=4096, 
        frame_step=62,
        switch=False,
        permute=False,
        permute_type=None,
        architecture='resindep'
    ):
        super().__init__()
        # Architecture type
        self._architecture = architecture

        # Sub-networks
        self.PHN = PrepHidingNet(self._architecture, transform)
        self.RN = RevealNet()

        # STDCT parameters
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.switch = switch if transform == 'cosine' else False

        # Noise parameters
        self.add_noise = add_noise
        self.noise_kind = noise_kind
        self.noise_amplitude = noise_amplitude

        #stft initialization
        if (self.add_noise) and (self.noise_kind is not None) and (self.noise_amplitude is not None) and transform == 'fourier':
            self.stft = STFT(filter_length=2 ** 11 - 1,
                        hop_length=132,
                        win_length=2 ** 11 - 1,
                        window='hann').cuda()
            self.stft.num_samples = 67522
        self.transform = transform

        # permute parameters
        self.i = 0
        self.permute = permute
        self.permute_type = permute_type

    def forward(self, secret, cover, phase): #noise phase
        # Create a new channel with 0 (R,G,B) -> (R,G,B,0)
        zero = torch.zeros(1, 1, 256, 256).type(torch.float32).cuda()
        secret = torch.cat((secret,zero),1)
        
        if (self._architecture == 'plaindep') or (self._architecture == 'resdep'):
            hidden_signal = self.PHN(secret, cover)
        else:
            hidden_signal = self.PHN(secret)
        if self.i==0 and self.permute_type == 'FIX' and self.permute:
            #permute parameters
            self.idx_fix = fix_permutation_idx(hidden_signal)
            self.i = 1000

        if self.permute and self.permute_type == 'FIX':
            # using fix permutation
            hidden_signal = fix_permutation(hidden_signal,self.idx_fix)

        # Hidden signal permutation
        elif self.permute and self.permute_type == 'DIFF':
            hidden_signal, idx = permutation(hidden_signal)

        # Residual connection
        container = cover + hidden_signal if (self._architecture != 'plaindep') else hidden_signal

        # inverse permutation
        if self.permute and self.permute_type == 'DIFF':
            container_rn = unpermutation(container, idx)
        elif self.permute and self.permute_type == 'FIX':
            container_rn = unpermutation(container, self.idx_fix)

        if (self.add_noise) and (self.noise_kind is not None) and (self.noise_amplitude is not None):


            if self.transform == 'cosine':
                # Switch domain
                container_wav =  isdct_torch(
                    container.squeeze(0).squeeze(0),
                    frame_length=self.frame_length,
                    frame_step=self.frame_step,
                    window=torch.hamming_window
                )
                # Generate spectral noise
                noise = add_noise(
                    container_wav,
                    self.noise_kind[np.random.randint(0, len(self.noise_kind))],
                    self.noise_amplitude[np.random.randint(0, len(self.noise_amplitude))]
                ).type(torch.float32)
                spectral_noise = sdct_torch(
                    noise,
                    frame_length=self.frame_length,
                    frame_step=self.frame_step
                ).unsqueeze(0).cuda()
            else:
                # Switch domain
                container_wav = self.stft.inverse(container.squeeze(1).cuda(), phase.squeeze(1).cuda())
                # Generate spectral noise
                noise = add_noise(
                        container_wav,
                        self.noise_kind,
                        self.noise_amplitude
                ).type(torch.float32)
                spectral_noise = self.stft.transform(noise)[0].unsqueeze(0).cuda()

            # Add noise in frequency
            corrupted_container = container + spectral_noise
            
            # Reveal image
            revealed = self.RN(corrupted_container)
        else: 
            if self.switch:
                # Switch domain and back
                container_wav =  isdct_torch(
                    container.squeeze(0).squeeze(0),
                    frame_length=self.frame_length, 
                    frame_step=self.frame_step, 
                    window=torch.hamming_window
                )
                container = sdct_torch(
                    container_wav,
                    frame_length=self.frame_length, 
                    frame_step=self.frame_step
                ).unsqueeze(0).unsqueeze(0)

            # Reveal image
            if self.permute:
                revealed = self.RN(container_rn)
            else:
                revealed = self.RN(container)

        return container, revealed