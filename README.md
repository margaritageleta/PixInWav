# PixInWav: Residual Steganography for Hiding Pixels in Audio

This repository includes a python implemenation of `StegoUNet`, a deep neural network modelling an audio steganographic function. 

> Steganography comprises the mechanics of hiding secret data within a cover media which may be publicly available with the main premise that the fact that the communication is  taking  place  is  hidden  as  well. 

![alt text](front/img/example.png "Example")

If you find this paper or implementation useful, please consider citing our [ICASSP paper](https://ieeexplore.ieee.org/document/9746191):
```{tex}
@INPROCEEDINGS{geleta2021pixinwav,  
      author={Geleta, Margarita and Puntí, Cristina and McGuinness, Kevin and Pons, Jordi and Canton, Cristian and Giro-i-Nieto, Xavier},  
      booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},   
      title={Pixinwav: Residual Steganography for Hiding Pixels in Audio},   
      year={2022},  volume={},  number={},  
      pages={2485-2489},  
      doi={10.1109/ICASSP43922.2022.9746191}
}
```
And/or the [ArXiv preprint](https://arxiv.org/abs/2106.09814):
```{tex}
@misc{geleta2021pixinwav,
      title={PixInWav: Residual Steganography for Hiding Pixels in Audio}, 
      author={Margarita Geleta and Cristina Punti and Kevin McGuinness and Jordi Pons and Cristian Canton and Xavier Giro-i-Nieto},
      year={2021},
      eprint={2106.09814},
      archivePrefix={arXiv},
      primaryClass={cs.MM}
}
```

## Repository outline

In the `src` folder we find:

- `umodel.py`: the complete audio steganography model with RGB or B&W  images as input.
- `loader.py`: the loader script to create the customized dataset from RGB or B&W image (ImageNet) + audio.
- `trainer_rgb.py`: a script to either train a model from scratch using provided training data or loading a pre-trained `StegoUNet` model for RGB or B&W images.
- `losses.py`: a script with all the losses and metrics defined for training. Uses a [courtesy script](https://github.com/Po-Hsun-Su/pytorch-ssim) to compute the SSIM metric.
- `pystct.py`: [courtesy script](https://github.com/jonashaag/pydct) to perform Short-Time Cosine Transform on raw audio waveforms.
- `pydtw.py`: [courtesy script](https://github.com/Sleepwalking/pytorch-softdtw) to compute SoftDTW as an additional term in the loss function.

In the `scripts` folder we find:

- `train.sh`: a sample `sbatch` script for Slurm used for sending training jobs.

## Dependencies

First, create a virtual environment on your local repository and activate it:
```
$ python3 -m venv env
$ source env/bin/activate
```
The dependencies are listed in `requirements.txt`. Note that you need [PyTorch](https://pytorch.org) v1.7.1 and [TorchAudio](torchaudio) v0.7.2. With `pip` installed, just run:
```
$ (env) pip3 install -r requirements.txt
```

## Data
We use [ImageNet](http://image-net.org) (ILSVRC2012) 10,000 images for training and 900 images for validation. Regarding audio, we use [FSDNoisy18K](http://www.eduardofonseca.net/FSDnoisy18k/) which has 17584 audios for training and 946 audios for validation. Each audio has a different duration, in our case we sample randomly different sections of audios that correspond to 1.5 seconds approximately (67522 samples). 

## Usage
After the installation of the requirements, to execute the `trainer_rgb.py` script, do:
```
$ (env) srun -u --gres=gpu:2,gpumem:12G 
        -p gpi.compute 
        --time 23:59:59 
        --mem 50G python3 trainer_rgb.py 
        --beta [beta_value] 
        --lr [learning_rate_value] 
        --summary "[description_of_the_run]" 
        --experiment [experiment_number]
        --add_noise [True/False]
        --noise_kind [gaussian/speckle/salt/pepper/salt&pepper]
        --noise_amplitude [float]
        --add_dtw_term [True/False]
        --rgb [use_rgb_or_b&w_images]
        --transform [cosine/fourier]
        --on_phase [if_fourier_hide_on_magnitude_or_phase]
        --architecture [resindep/resdep/resscale/plaindep]
```
Reserve as minimum 12G of GPU memory per GPU, otherwise you may be CUDA OOM. Or, run the `sbatch` script as follows:
```
$ (env) ./train.sh [experiment_number]
```
Defining all the arguments and hyperparameters in the script beforehand.

### Loss function and optimization
+ `--lr` defined the learning rate of the Adam optimizer.
+ `--beta` determines the beta parameter of the loss function, refer to the paper for details. 
+ `--add_dtw_term` allows adding an additional term to the loss function. Adding it has shown improvements, refer to the paper for details.
### Model architecture and constraints
+ With `--rgb` you can choose to train on RGB or B&W images.
+ `--architecture` allows to change the underlying architecture. It lists the 4 types of model explained in the paper, refer to it for more details.
+ With `--transform` you can change the transform to obtain the audio spectrogram. Available transforms include STDCT (Short-Time Discrete Cosine Transform Type II) and STFT (Short-Time Fourier). 
+ If you use STFT, you can choose to hide the image in the magnitude or in the phase. You can control thos behaviour with `--on_phase`.
### Noise addition
+ For increasing the robustness of the steganographic function, you can add noise into the audio during training time with `--add_noise`.
+ If you `--add_noise` then you should choose the `--noise_kind` and `--noise_amplitude`.

### Monitor the training process
By default, `wandb` checkpoints are created when you execute the `trainer_rgb.py` script (you should login into your [wandb](https://wandb.ai) account first). This allows tracking the learning curves in the web application.

If you prefer using `tensorboard` checkpoints, you will need to install `tensorboardX` and add the needed lines of code to save the values. Once it is done, just run in another shell window:
```
$ (env) tensorboard dev upload --logdir 'logs/[timestamp]'
```
Where `logs` is the directory you choose to store your logs.

### Training from a checkpoint
To train a model from a checkpoint, follow these steps in the `main` function in `trainer_rgb.py`:
```
## Load the checkpoint
chk = torch.load('[checkpoint_path]/[checkpoint_name].pt', map_location='cpu')
model = StegoUNet()
model = nn.DataParallel(model)
## Load the weights into the model
model.load_state_dict(chk['state_dict'])

[...]

train(
    model=model, 
    tr_loader=train_loader, 
    vd_loader=test_loader, 
    beta=float(args.beta), 
    lr=float(args.lr), 
    epochs=15, 
    slide=15,
    prev_epoch=chk['epoch'], ## Specify this!
    prev_i=chk['i'], ## Specify this!
    summary=args.summary,
    experiment=int(args.experiment)
)
```

## License

**NOTICE**: This software is available for use free of charge for academic research use only. Commercial users, for profit companies or consultants, and non-profit institutions not qualifying as *academic research* must contact `geleta@berkeley.edu` for a separate license. 
