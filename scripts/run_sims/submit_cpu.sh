#!/bin/bash

#SBATCH --job-name=run_dataset
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --output=/scratch/cp3759/pyqg_data/sims/torchqg_sims/slurm_dump/slurm_%j.out
#SBATCH --error=/scratch/cp3759/pyqg_data/sims/torchqg_sims/slurm_dump/slurm_%j.err
#SBATCH --array=1-275


# Begin execution
module purge

singularity exec --nv \
	    --overlay /scratch/cp3759/sing/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python3 /home/cp3759/Projects/torch_qg/scripts/run_sims/run_forcing_sim.py --save_to /scratch/cp3759/pyqg_data/sims/every_snap.nc --save_to /scratch/cp3759/pyqg_data/sims/torchqg_sims/0_step/ --run_number $SLURM_ARRAY_TASK_ID"