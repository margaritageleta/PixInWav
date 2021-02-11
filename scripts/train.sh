#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=PixInWav-$1
#SBATCH -p gpi.compute
#SBATCH -c 4
#SBATCH --gres=gpu:1,gpumem:16G
#SBATCH --mem=80G
#SBATCH --time=23:59:59
#SBATCH -o logs/exp_$1.log
#SBATCH -e logs/exp_$1.err

echo "Executing experiment $1"
python3 src/trainer.py
echo "Success!"
EOT
