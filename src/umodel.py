import torch
import numpy as np
import torch.nn as nn
from torch import utils
import torch.nn.functional as F
from pystct import sdct_torch, isdct_torch
from noises import add_noise

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

    def __init__(self, in_channels, out_channels, mid_channels=None, image_alone = False, magphase = False):
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
        convinput = out_channels * 2 if self.image_alone else out_channels * 3
        if magphase: convinput = 3

        self.conv = DoubleConv(convinput, out_channels, mid_channels)

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
            im_enc = [nn.Upsample(scale_factor=(2, 1), mode='bilinear', align_corners=True)(im)]
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
    def __init__(self, phase_type=None):
        super(RevealNet, self).__init__()

        self.phase_type=phase_type
        self.pixel_unshuffle = PixelUnshuffle(2)

        # If phase_type == RN, modify RevealNet to accept 2 channels as input instead of 1
        if phase_type != 'RN':
            self.im_encoder_layers = nn.ModuleList([
                Down(1, 64),
                Down(64, 64 * 2)
            ])
        else:
            self.im_encoder_layers = nn.ModuleList([
                Down(2, 64),
                Down(64, 64 * 2)
            ])

        if phase_type != 'RN':
            self.im_decoder_layers = nn.ModuleList([
                Up(64 * 2, 64, image_alone=True),
                Up(64, 1, image_alone=True)
            ])
        else:        
            self.im_decoder_layers = nn.ModuleList([
                Up(64 * 2, 64, image_alone=True),
                Up(64, 1, image_alone=True, magphase=True)
            ])

        # If phase_type == '2D' or '3D', create the conv kernel to merge afterwards
        if phase_type == '2D':
            self.mag_phase_join = nn.Conv2d(6,3,1)
        elif phase_type == '3D':
            self.mag_phase_join = nn.Conv3d(2,1,1)

    def forward(self, ct, ct_phase=None):

        # ct is not None if and only if phase_type != None
        # If using the phase only, 'ct' is the phase, else the magnitude
        assert not (self.phase_type is None and ct_phase is not None)
        assert not (self.phase_type is not None and ct_phase is None)

        im_enc = [F.interpolate(ct, size=(256 * 2, 256 * 2))]
        if ct_phase is not None:
            im_enc_phase = [F.interpolate(ct_phase, size=(256 * 2, 256 * 2))]

        if self.phase_type == 'RN':
            # Concatenate mag and phase containers to input to RevealNet
            im_enc = [torch.cat((im_enc[0], im_enc_phase[0]), 1)]

        for enc_layer_idx, enc_layer in enumerate(self.im_encoder_layers):
            im_enc.append(enc_layer(im_enc[-1]))
            if self.phase_type == 'mean' or self.phase_type == '2D' or self.phase_type == '3D':
                im_enc_phase.append(enc_layer(im_enc_phase[-1]))

        
        im_dec = [im_enc.pop(-1)]
        if self.phase_type == 'mean' or self.phase_type == '2D' or self.phase_type == '3D':
            im_dec_phase = [im_enc_phase.pop(-1)]

        for dec_layer_idx, dec_layer in enumerate(self.im_decoder_layers):
            im_dec.append(
                dec_layer(im_dec[-1], 
                im_enc[-1 - dec_layer_idx])
            )
            if self.phase_type == 'mean' or self.phase_type == '2D' or self.phase_type == '3D':
                im_dec_phase.append(
                    dec_layer(im_dec_phase[-1], 
                    im_enc_phase[-1 - dec_layer_idx])
                )
        
        # Pixel Unshuffle and delete 4th component
        revealed = torch.narrow(self.pixel_unshuffle(im_dec[-1]), 1, 0, 3)

        if self.phase_type == 'mean' or self.phase_type == '2D' or self.phase_type == '3D':
            # Postprocess phase
            revealed_phase = torch.narrow(self.pixel_unshuffle(im_dec_phase[-1]), 1, 0, 3)

            # Join with magnitude
            if self.phase_type == 'mean':
                return revealed.add(revealed_phase)*0.5
            elif self.phase_type == '2D':
                join = torch.cat((revealed,revealed_phase),1)
                return self.mag_phase_join(join)
            elif self.phase_type == '3D':
                revealed = revealed.unsqueeze(0)
                revealed_phase = revealed_phase.unsqueeze(0)
                join = torch.cat((revealed,revealed_phase),1)
                return self.mag_phase_join(join).squeeze(1)
        else:
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
        architecture='resindep',
        phase_type=None
    ):
        super().__init__()
        # Architecture type
        self._architecture = architecture

        # Sub-networks
        self.PHN = PrepHidingNet(self._architecture, transform)
        self.RN = RevealNet(phase_type)

        # STDCT parameters
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.switch = switch if transform == 'cosine' else False

        # Noise parameters
        self.add_noise = add_noise if transform == 'cosine' else False
        self.noise_kind = noise_kind
        self.noise_amplitude = noise_amplitude

        # Phase parameters
        self.phase_type=phase_type # phase_type != None implies not on_phase

    def forward(self, secret, cover, cover_phase=None):

        # cover_phase is not None if and only if phase_type != None
        # If using the phase only, 'cover' is actually the phase!
        assert not (self.phase_type is None and cover_phase is not None)
        assert not (self.phase_type is not None and cover_phase is None)

        # Create a new channel with 0 (R,G,B) -> (R,G,B,0)
        zero = torch.zeros(1, 1, 256, 256).type(torch.float32).cuda()
        secret = torch.cat((secret,zero),1)
        
        if (self._architecture == 'plaindep') or (self._architecture == 'resdep'):
            hidden_signal = self.PHN(secret, cover)
        else:
            hidden_signal = self.PHN(secret)

        # Residual connection
        container = cover + hidden_signal if (self._architecture != 'plaindep') else hidden_signal
        if cover_phase is not None:
            container_phase = cover_phase + hidden_signal if (self._architecture != 'plaindep') else hidden_signal

        if (self.add_noise) and (self.noise_kind is not None) and (self.noise_amplitude is not None):
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
            if cover_phase is not None:
                revealed = self.RN(container, container_phase)
            else:
                revealed = self.RN(container)

        return container, revealed
