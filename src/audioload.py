import numpy as np
import glob as glob
import torchaudio
import scipy.fft
import torch
import torch_dct
import random
from torch_stft import STFT

folder = '/mnt/gpid08/users/teresa.domenech/train'

#transform parameters
limit = 67522  # 2 ** 16 + 2 ** 11 - 2 ** 6 + 2
#frame_length = 2 ** 12
#frame_step = 2 ** 6 - 2
frame_length = 2 ** 11 - 1
frame_step = 132

def sdct_torch(signals, frame_length, frame_step, window=torch.hamming_window):
    """Compute Short-Time Discrete Cosine Transform of `signals`.
    No padding is applied to the signals.
    Parameters
    ----------
    signal : Time-domain input signal(s), a `[..., n_samples]` tensor.
    frame_length : Window length and DCT frame length in samples.
    frame_step : Number of samples between adjacent DCT columns.
    window : Window to use for DCT.  Either a window tensor (see documentation for `torch.stft`),
        or a window tensor constructor, `window(frame_length) -> Tensor`.
        Default: hamming window.
    Returns
    -------
    dct : Real-valued F-T domain DCT matrix/matrixes, a `[..., frame_length, n_frames]` tensor.
    """
    framed = signals.unfold(-1, frame_length, frame_step)
    if callable(window):
        window = window(frame_length).to(framed)
    if window is not None:
        framed = framed * window
    return torch_dct.dct(framed, norm="ortho").transpose(-1, -2)

if __name__ == '__main__':
    audios_mag = []
    audios_phase = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #fourier initialization
    stft = STFT(
        filter_length=2 ** 11 - 1,
        hop_length=132,
        win_length=2 ** 11 - 1,
        window='hann'
    )
    stft.num_samples = 67522

    for audio_path in glob.glob(f'{folder}/*.wav'):
        sound, sr = torchaudio.load(audio_path)
        # Get the samples dimension
        sound = sound[0]
        # Create a temporary array
        tmp = torch.zeros([limit, ]).normal_(mean=0, std=0.005)
        # Cut the audio on limit
        if sound.numel() < limit:
            tmp[:sound.numel()] = sound[:]
        else:
            i = random.randint(0, len(sound) - limit)
            tmp[:] = sound[i:i + limit]
        # cosine transform
        '''
        data_cosine = sdct_torch(
            tmp.type(torch.float32),
            frame_length=frame_length,
            frame_step=frame_step).to(device)
        audios.append(data_cosine.cpu().numpy())
        '''
        # fourier transform
        magnitude, phase = stft.transform(tmp.unsqueeze(0).type(torch.float32))
        audios_mag.append(magnitude.cpu().numpy())
        audios_phase.append(phase.cpu().numpy())

    # save array in .npy
    print('finalize')
    np.save('/mnt/gpid08/users/teresa.domenech/audio_fourier_train', audios_mag)
    np.save('/mnt/gpid08/users/teresa.domenech/audio_fourier_train_phase', audios_phase)
