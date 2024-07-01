#!/bin/bash 
#SBATCH --job-name=ES6_test       # Job name
#SBATCH --mail-user=ztushar1@umbc.edu       # Where to send mail
#SBATCH --mem=45000                       # Job memory request
#SBATCH --gres=gpu:1                     # Number of requested GPU(s) 
#SBATCH --time=72:00:00                   # Time limit days-hrs:min:sec
#SBATCH --constraint=rtx_8000            # Specific hardware constraint
#SBATCH --error=aada_files/ES6test.err                # Error file name
#SBATCH --output=aada_files/ES6test.out               # Output file name
# sza [60.0,40.0,20.0,4.0]
# vza [60,30,15,0,-15,-30,-60]
export CUDA_LAUNCH_BLOCKING=1
python ES6_cloud_generate.py