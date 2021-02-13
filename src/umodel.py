import torch
from torch import utils
import torch.nn as nn
import torch.nn.functional as F

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
        # print(8*'='+'DOUBLE CONV')
        # print(8*'-'+f'Starting shape ({x.shape[1]}x{x.shape[2]}x{x.shape[3]})')
        x = self.conv1(x)
        x = self.conv2(x)
        # print(8*'-'+f'Final shape ({x.shape[1]}x{x.shape[2]}x{x.shape[3]})')
        return x


class Down(nn.Module):

    def __init__(self, in_channels, out_channels, downsample_factor=8, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels
        self.conv = DoubleConv(in_channels, out_channels, mid_channels)
        self.down = nn.MaxPool2d(downsample_factor)

    def forward(self, x):
        # print(5*'='+'DOWN')
        # print(5*'-'+f'Original shape ({x.shape[1]}x{x.shape[2]}x{x.shape[3]})')
        x = self.conv(x)
        x = self.down(x)
        # print(5*'-'+f'Downsampled shape ({x.shape[1]}x{x.shape[2]}x{x.shape[3]})')
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
        # print(5*'='+'UP')
        # print(5*'-'+f'Original shape ({mix.shape[1]}x{mix.shape[2]}x{mix.shape[3]})')
        mix = self.up(mix)
        # print(5*'-'+f'Upsampled shape ({mix.shape[1]}x{mix.shape[2]}x{mix.shape[3]})')
        x = torch.cat((mix, im_opposite), dim=1) if self.image_alone else torch.cat((au_opposite, mix, im_opposite), dim=1)
        # print(5*'-'+f'Concat shape ({x.shape[1]}x{x.shape[2]}x{x.shape[3]})')
        return self.conv(x)


class PrepHidingNet(nn.Module):
    def __init__(self):
        super(PrepHidingNet, self).__init__()

        self.au_encoder_layers = nn.ModuleList([
            Down(1, 64),
            Down(64, 64 * 2)
        ])

        self.im_encoder_layers = nn.ModuleList([
            Down(1, 64),
            Down(64, 64 * 2)
        ])

        self.decoder_layers = nn.ModuleList([
            Up(64 * 2, 64),
            Up(64, 1)
        ])

    def forward(self, im, au):

        im_enc = [nn.Upsample(scale_factor=(16, 4), mode='bilinear', align_corners=True)(im)]
        au_enc = [au]

        for enc_layer_idx, enc_layer in enumerate(self.au_encoder_layers):
            # print(f'au Encoder layer #{enc_layer_idx + 1}')
            au_enc.append(enc_layer(au_enc[-1]))

        for enc_layer_idx, enc_layer in enumerate(self.im_encoder_layers):
            # print(f'im Encoder layer #{enc_layer_idx + 1}')
            im_enc.append(enc_layer(im_enc[-1]))

        mix_dec = [au_enc.pop(-1) + im_enc.pop(-1)]

        for dec_layer_idx, dec_layer in enumerate(self.decoder_layers):
            # print(f'Decoder layer #{dec_layer_idx + 1}')
            mix_dec.append(dec_layer(mix_dec[-1], im_enc[-1 - dec_layer_idx], au_enc[-1 - dec_layer_idx]))

        return mix_dec[-1]


class RevealNet(nn.Module):
    def __init__(self):
        super(RevealNet, self).__init__()

        self.au_encoder_layers = nn.ModuleList([
            Down(1, 64),
            Down(64, 64 * 2)
        ])

        self.im_encoder_layers = nn.ModuleList([
            Down(1, 64),
            Down(64, 64 * 2)
        ])

        self.decoder_layers = nn.ModuleList([
            Up(64 * 2, 64, image_alone=True),
            Up(64, 1, image_alone=True)
        ])

    def forward(self, ct):

        im_enc = [F.interpolate(ct, size=(256, 256))]

        for enc_layer_idx, enc_layer in enumerate(self.im_encoder_layers):
            # print(f'im Encoder layer #{enc_layer_idx + 1}')
            im_enc.append(enc_layer(im_enc[-1]))

        im_dec = [im_enc.pop(-1)]

        for dec_layer_idx, dec_layer in enumerate(self.decoder_layers):
            # print(f'Decoder layer #{dec_layer_idx + 1}')
            im_dec.append(dec_layer(im_dec[-1], im_enc[-1 - dec_layer_idx]))

        return im_dec[-1]


class StegoUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.PHN = PrepHidingNet()
        self.RN = RevealNet()

    def forward(self, secret, cover):
        # print('Process + Hiding Network working ...')
        hidden_signal = self.PHN(secret, cover)
        # print('Pre Process ...')
        container = cover + hidden_signal

        # print('Reveal Network working ...')
        revealed = self.RN(container)
        # print('DONE! ...')
        
        return container, revealed