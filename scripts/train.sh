#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=$1PixInWav
#SBATCH -p gpi.compute
#SBATCH -c 4
#SBATCH --gres=gpu:2,gpumem:12G
#SBATCH --mem=32G
#SBATCH --time=23:59:59
#SBATCH -o /dev/null
#SBATCH -e $OUT_PATH/logs/pixinwav_$1.err

echo "Executing experiment $1"
python3 ../src/trainer_rgb.py --beta $2 --lr $3 --summary "RGB unshuffle DTW + noise, beta=$2, lr=$3" --experiment $1
echo "Success!"
EOT
