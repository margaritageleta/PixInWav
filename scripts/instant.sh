# Usage:  ./instant.sh [experiment_number] [beta] [learning_rate]
# Sample: ./instant.sh 13 0.05 0.001  

echo "Executing experiment $1"
srun -u -c 4 -p gpi.develop -w gpic13 --time 00:30:59 --mem 32G --gres=gpu:1,gpumem:15G python3 ../src/trainer_rgb_1.py --beta $2 \
--lr $3 \
--summary "Run $1: Try, beta=$2, lr=$3" \
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