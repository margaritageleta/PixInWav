#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=$1-PixInWav
#SBATCH -p gpi.compute
#SBATCH -c 4
#SBATCH --gres=gpu:2,gpumem:16G
#SBATCH --mem=80G
#SBATCH --time=23:59:59
#SBATCH -o /dev/null
#SBATCH -e /dev/null

echo "Executing experiment $1"
python3 ../src/trainer.py
echo "Success!"
EOT
