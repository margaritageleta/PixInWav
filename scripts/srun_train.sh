#!/bin/bash

beta=0.75
lr=0.001
experiment=707
summary="DemoMP"
output="Outputs/$summary-$experiment.txt"
add_noise="False"
noise_kind="None"
noise_amplitude=0
add_l1_term="True"
rgb="True"
from_checkpoint="False"
transform="fourier"
on_phase="True"
phase_type="RN"
architecture="resindep"


srun -u --gres=gpu:2,gpumem:12G -p gpi.compute -o $output --time 23:59:59 --mem 50G python3 ~/PixInWav/src/trainer_rgb.py --beta $beta --lr $lr --summary $summary --experiment $experiment --from_checkpoint $from_checkpoint --add_noise $add_noise --noise_kind $noise_kind --noise_amplitude $noise_amplitude --add_l1_term $add_l1_term --rgb $rgb --transform $transform --on_phase $on_phase --phase_type $phase_type --architecture $architecture

#srun -u --gres=gpu:2,gpumem:12G -p gpi.compute -o $output --time 23:59:59 --mem 50G python3 ~/PixInWav/src/DataParallelFix.py --beta $beta --lr $lr --summary $summary --experiment $experiment --from_checkpoint $from_checkpoint --add_noise $add_noise --noise_kind $noise_kind --noise_amplitude $noise_amplitude --add_dtw_term $add_dtw_term --rgb $rgb --transform $transform --on_phase $on_phase --architecture $architecture
