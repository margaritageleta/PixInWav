# PixInWav: Hiding pixels with Deep Steganography

This repository includes a python implemenation of `StegoNet`, a deep neural network modelling an audio steganographic function. 

In the `src` folder we find:

- `model.py`: the best-performing audio steganography model.
- `loader.py`: the loader script to create the customized dataset from image (ImageNet) + audio.
- `pystct.py`: courtesy script to perform Short-Time Cosine Transform on raw audio waveforms.
- `trainer.py`: a script to either train a model from scratch using provided training data or loading a pre-trained `StegoNet` model.

## Dependencies

First, create a virtual environment on your local repository and activate it:
```
$ python3 -m venv env
$ source env/bin/activate
```
The dependencies are listed in `requirements.txt`. With `pip` installed, just run:
```
$ (env) pip3 install -r requirements.txt
```

## Usage
To execute the `trainer.py` script, do:
```
srun -u --gres=gpu:2,gpumem:10G -p gpi.compute --time 14:59:00 --mem 50G python trainer.py
```
Reserve as minimum 35G of memory, otherwise you will be OOM.

By default, `tensorboard` checkpoints are created when you execute the `trainer.py` script. To track the learning curves, run in another shell window:
```
tensorboard dev upload --logdir 'logs/[timestamp]'
```

To train a model from a checkpoint, follow these steps in the `main` function in `trainer.py`:
```
chk = torch.load('checkpoints/[checkpoint_name].pt', map_location='cpu')
model = StegoNet()
model.load_state_dict(chk['state_dict'])
[...]
train(train_loader, beta = 0.3, lr = 0.001, epochs = 5, prev_epoch = chk['epoch'], prev_i = chk['i'])
```

Different reconstruction losses can be used to train the network. To use the waveform reconstruction loss, please use `stego_loss_wav`. To use spectrogram reconstruction loss, use `stego_loss`.

## License

**NOTICE**: This software is available for use free of charge for academic research use only. Commercial users, for profit companies or consultants, and non-profit institutions not qualifying as *academic research* must contact `rita.geleta@jediupc.com` for a separate license. 




