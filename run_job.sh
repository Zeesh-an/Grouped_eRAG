#!/bin/bash
#SBATCH --job-name=run_demo_job
#SBATCH --output=demo_output.log
#SBATCH --error=demo_error.log
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# Load modules or activate your virtual environment
# module load python/3.10 cuda/11.7
source /local/scratch3/zmemon/venv/bin/activate
export TRANSFORMERS_CACHE=/local/scratch3/zmemon/hf_model_cache
echo "=== SLURM_JOB_ID: $SLURM_JOB_ID ==="
echo "=== Hostname: $(hostname) ==="

# Start GPU usage monitoring in background
while true; do
    echo "==== $(date) ====" >> gpu_usage.log
    nvidia-smi >> gpu_usage.log
    sleep 5
done &
GPU_MONITOR_PID=$!

# Run your Python GPU script
echo "=== Running Scripts ==="
# python load_data.py
# python model.py
python fid_full_rag.py

# Stop the GPU usage logging
kill $GPU_MONITOR_PID
echo "=== GPU Monitoring stopped ==="