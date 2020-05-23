#!/bin/bash
#
#SBATCH -J jupyter                    # Job name
#SBATCH -o jupyter.out                # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu-shared                 # Queue name
#SBATCH --gres=gpu:1
#SBATCH -N 1                          # Total number of nodes requested (20 cores/node)
#SBATCH -n 1                          # Total number of mpi tasks requested
#SBATCH -t 48:00:00                   # Run time (hh:mm:ss) - 4 hours
#SBATCH -A sio134


ibrun -n 1 python local_distributor.py --trial_id_min SHERPA_TRIAL_ID_MIN --trial_id_max SHERPA_TRIAL_ID_MAX --output_dir SHERPA_OUTPUT_DIR &

wait
