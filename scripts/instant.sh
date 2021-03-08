# Usage:  ./instant.sh [experiment_number] [beta] [learning_rate]
# Sample: ./instant.sh 13 0.05 0.001  

echo "Executing experiment $1"
srun -u -c 1 -p gpi.develop --time 02:00:00 --mem 32G --gres=gpu:2,gpumem:12G python3 ../src/trainer_rgb.py --beta $2 \
--lr $3 \
--summary "Salt, beta=$2, lr=$3" \
--experiment $1 \
--add_noise True \
--noise_kind "salt" \
--noise_amplitude 0.001
echo "Success!"