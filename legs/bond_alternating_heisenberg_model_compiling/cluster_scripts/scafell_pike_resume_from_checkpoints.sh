#!/bin/sh
#BSUB -q scafellpikeSKL
#BSUB -W 48:00
#BSUB -o out-%J.o
#BSUB -e err-%J.e
#BSUB -R "span[ptile=32]"
#BSUB -n 32

module load python3/3.12
source /lustre/scafellpike/local/HT07737/gxp02/gxp97-gxp02/legs_env/bin/activate

module load parallel

export OMP_NUM_THREADS=1   # Number of threads per python process

checkpoint_dir=()
delete_prev_chkpt=("True")
log_dir=("./logs_J1_-2.0/")


command="parallel --delay 0.2 -j 2"     # -j N means N jobs at a time run in parallel
command+=" python resume_from_checkpoint.py"
command+=" -cd {1}"
command+=" -dpc {2}"
command+=" -ld {3}"

$command \
::: ${checkpoint_dir[@]} \
::: ${delete_prev_chkpt[@]} \
::: ${log_dir[@]}