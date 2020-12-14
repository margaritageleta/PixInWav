import pathlib
import torch
import glob as glob
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio
import librosa
import librosa.display
import scipy.fft
import time
import warnings
warnings.filterwarnings("ignore")

MY_FOLDER = '/mnt/gpid07/imatge/margarita.geleta/pix2wav'
IMAGE_FOLDER = '/projects/deep_learning/ILSVRC/ILSVRC2012'
AUDIO_FOLDER = f'{MY_FOLDER}/data/FSDnoisy18k.audio_train'

def sdct(signal, frame_length, frame_step, window="hamming"):
    """Compute Short-Time Discrete Cosine Transform of `signal`.
    Parameters
    ----------
    signal : Time-domain input signal of shape `(n_samples,)`.
    frame_length : Window length and DCT frame length in samples.
    frame_step : Number of samples between adjacent DCT columns.
    window : Window specification passed to ``librosa.filters.get_window``.
        Default: "hamming".  Window to use for DCT.
    Returns
    -------
    dct : Real-valued F-T domain DCT matrix of shape `(frame_length, n_frames)`
    """
    framed = librosa.util.frame(signal, frame_length, frame_step)
    if window is not None:
        window = librosa.filters.get_window(window, frame_length, fftbins=True).astype(
            signal.dtype
        )
        framed = framed * window[:, np.newaxis]
    return scipy.fft.dct(framed, norm="ortho", axis=-2)

def isdct(dct, *, frame_step, frame_length=None, window="hamming"):
    """Compute Inverse Short-Time Discrete Cosine Transform of `dct`.
    Parameters other than `dct` are keyword-only.
    Parameters
    ----------
    dct : DCT matrix from `sdct`.
    frame_step : Number of samples between adjacent DCT columns (should be the
        same value that was passed to `sdct`).
    frame_length : Ignored. Window length and DCT frame length in samples.
        Can be None (default) or same value as passed to `sdct`.
    window : Window specification passed to ``librosa.filters.get_window``.
        Default: "hamming".  Window to use for IDCT.
    Returns
    -------
    signal : Time-domain signal reconstructed from `dct` of shape `(n_samples,)`.
        Note that `n_samples` may be different from the original signal's length as passed to `sdct`.
    """
    frame_length2, n_frames = dct.shape
    assert frame_length in {None, frame_length2}
    signal = overlap_add(scipy.fft.idct(dct, norm="ortho", axis=-2), frame_step)
    if window is not None:
        window = librosa.filters.get_window(window, frame_length2, fftbins=True).astype(
            dct.dtype
        )
        window_frames = np.tile(window[:, np.newaxis], (1, n_frames))
        window_signal = overlap_add(window_frames, frame_step)
        signal = signal / window_signal
    return signal

def overlap_add(framed, frame_step):
    """Overlap-add ("deframe") a framed signal.
    Parameters
    ----------
    framed : array_like, frames of shape `(..., frame_length, n_frames)`.
    frame_step : Overlap to use when adding frames.
    Returns
    -------
    deframed : Overlap-add ("deframed") signal.
        np.ndarray of shape `(..., (n_frames - 1) * frame_step + frame_length)`.
    """
    *shape_rest, frame_length, n_frames = framed.shape
    deframed_size = (n_frames - 1) * frame_step + frame_length
    deframed = np.zeros((*shape_rest, deframed_size), dtype=framed.dtype)
    for i in range(n_frames):
        pos = i * frame_step
        deframed[..., pos : pos + frame_length] += framed[..., i]
    return deframed

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, folder: str, extension: str = "wav"):
        self._data = pathlib.Path(root) / folder
        self.extension = extension
        # Define the index (length) of dataset
        # Define the a list with own indexes of files
        self._index = 0
        self._indices = []
        # Get audio files
        for audio in glob.glob(f'{AUDIO_FOLDER}/*'):
            self._indices.append(audio)
            self._index += 1
        # Corresponds to 1.5 seconds approximately
        self._limit = 2 ** 16 + 2 ** 11 - 1

    def __len__(self):
        return self._index

    def __getitem__(self, index):
        audio_path = self._indices[index]
        sound, sr = torchaudio.load(audio_path)
        # Get the samples dimension
        sound = sound[0]
        # Create a temporary array
        tmp = torch.zeros([self._limit,])
        # Cut the audio on limit
        if sound.numel() < self._limit:
            tmp[:sound.numel()] = sound[:]
        else:
            tmp[:] = sound[:self._limit]
        sound_stct = sdct(tmp.numpy().astype(np.float32), frame_length=2048, frame_step=512)
        return sound_stct

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=5,
                      stride=4,
                      padding=2),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=5,
                stride=4,
                padding=1,
                output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=1,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def save_checkpoint(state, is_best, filename=f'{MY_FOLDER}/checkpoints/checkpoint.pt'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")

if __name__ == '__main__':
    dataset = AudioDataset(AUDIO_FOLDER, '')
    print(f'Dataset length: {dataset.__len__()}')
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=64,
                                             shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_epochs = 25  
    batch_size = 64
    model = Autoencoder().to(device);
    distance = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=0)

    since = time.time()
    best_loss = np.inf
    for epoch in range(num_epochs):
        counter = 0
        for i, data in enumerate(dataloader):
            # print(data.shape)
            src_sctc = data.unsqueeze(1).to(device)
            # print(src_sctc.shape)
            # ===================forward=====================
            output = model(src_sctc)
            loss = distance(output.squeeze(1), src_sctc.unsqueeze(1))
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('='*(i + 1) + f'\nLoss: {loss / ((i + 1) * dataloader.batch_size)}')
        # ===================log========================
        epoch_loss = loss / len(dataloader)

        is_best = bool(epoch_loss.detach().cpu() > best_loss)

        best_loss = min(epoch_loss.detach().cpu(), best_loss)

        # Save checkpoint if is a new best
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_accuracy': best_loss,
        }, is_best)

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, epoch_loss))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    torch.save(model.state_dict(), f'{MY_FOLDER}/models/autoencoder_{batch_size}_{num_epochs}_run1.pt')

