#!/bin/bash

### BASH SCRIPT TO RUN SET OF EXPERIMENT GENERATED FROM EXPERIMENT FILE ON EDINBURGH CLUSTER ###

# Maximum number of nodes to use for the job
# #SBATCH --nodes=1

# Generic resources to use - typically you'll want gpu:n to get n gpus
#SBATCH --gres=gpu:2

# Megabytes of RAM required. Check `cluster-status` for node configurations
#SBATCH --mem=16000

#amount of tasks run in parallel on each node - this is just a test!
# #SBATCH --ntasks-per-node=1

# Number of CPUs to use. Check `cluster-status` for node configurations
#SBATCH --cpus-per-task=8

# Maximum time for the job to run, format: days-hours:minutes:seconds
#  #SBATCH --time=7-00:00:00

# Partition of the cluster to pick nodes from (check `sinfo`)
# #SBATCH --partition=PGR-Standard

# Any nodes to exclude from selection
# #SBATCH --exclude=charles[05,12-18]

source ~/.bashrc
set -e
echo "Setting up log files"
USER=s1686853
SCRATCH_DISK=/disk/scratch
log_path=${SCRATCH_DISK}/${USER}/planning_baselines
mkdir -p ${log_path}

echo "Initializing Conda Environment"
CONDA_NAME=env
conda activate ${CONDA_NAME}
#mkdir -p ${log_path}

echo "Running experiment command"
experiment_text_file=$1
COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" ${experiment_text_file}`"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"

echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"