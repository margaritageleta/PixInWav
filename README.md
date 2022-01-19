# Hiding Pixels in their Spoken Narratives

This is the code used for the "Hiding pixels in their spoken narratives" thesis submitted in the ETSETB-UPC.

You can find the code for use the Localized Narratives dataset using image-audio pairs and non using it, apply permutation in the hidden signal, apply noise using the STFT spectograms and the code to train in a more efficient way using numpy files for store the audio transforms.

## Repository outline

In the `src` folder we find:

- `audioload.py`: the script for storage all the STFT audio clips in a numpy file.
- `umodel.py`: the complete audio steganography model with RGB or B&W images as input.
- `loader_ln.py`: the loader script to create the customized dataset from RGB or B&W image (COCO) + audio (Localized Narratives).
- `loader_fast_ln_STFT.py`: the loader script to create the customized dataset from RGB or B&W image (COCO) + audio (Localized Narratives) using numpy file to load the audio.
- `loader_noisy.py`: the loader script to create the customized dataset from RGB or B&W image (ImageNet) + audio (FsdNoisy).
- `trainer_rgb_ln.py`: a script to either train a model from scratch using provided training data or loading a pre-trained StegoUNet model for RGB or B&W images for Localized narratives dataset.
- `trainer_rgb_noisy.py`: a script to either train a model from scratch using provided training data or loading a pre-trained StegoUNet model for RGB or B&W images for FsdNoisy dataset.
- `losses.py`: a script with all the losses and metrics defined for training. Uses a courtesy script to compute the SSIM metric.
- `pystct.py`: courtesy script to perform Short-Time Cosine Transform on raw audio waveforms.
- `pydtw.py`: courtesy script to compute SoftDTW as an additional term in the loss function.
- `noises.py`: a script to compute different types of noise like: AWGN, speckle etc...

In the `notebooks` folder we find:

- `resultats_tere.ipynb`: a notebook with all the experiments used for the thesis.

In the `scripts` folder we find:

- `train.sh`: a sample sbatch script for Slurm used for sending training jobs.
