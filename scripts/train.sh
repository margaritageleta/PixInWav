#!/bin/bash
#SBATCH -o /dev/null
#SBATCH -e $OUT_PATH/logs/pixinwav_$1.err
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=$1PixInWav
#SBATCH -p gpi.compute
#SBATCH -c 4
#SBATCH --gres=gpu:2,gpumem:12G
#SBATCH --mem=50G
#SBATCH --time=23:59:59

echo "Executing experiment $1"
python3 ../src/trainer_rgb_1.py --beta $2 \
--lr $3 \
--summary "Run $1: ResIndep, beta=$2, lr=$3" \
--experiment $1 \
--add_noise False \
--noise_kind None \
--noise_amplitude 0 \
--add_dtw_term True \
--rgb True \
--from_checkpoint False \
--transform fourier \
--on_phase False \
--architecture resindep \
--permute False \
--permute_type None
echo "Success!"
EOT
