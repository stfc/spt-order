#!/bin/sh
#BSUB -q scafellpikeSKL
#BSUB -W 24:00
#BSUB -o out-%J.o
#BSUB -e err-%J.e
#BSUB -R "span[ptile=32]"
#BSUB -n 64

module load python3/3.12
source # Python path

module load parallel

export OMP_NUM_THREADS=1   # Number of threads per python process

qubits=(100)
even_coupling=(1.0)
odd_coupling=(-2.0)
target_truncs=(1e-12)
compressed_bond_dims=(8)
layers=({3..6})
max_iter=(1_000_000)
sufficient_cost=(1e-2)
backend=("QUIMB")
optimiser_methods=("lbfgsb" "adam")
mps_truncation=(1e-8)
mps_bond_dim=(50)
checkpoint_dir=("./checkpoint_J1_-2.0/")
delete_prev_chkpt=("True")
log_dir=("./logs_J1_-2.0/")


command="parallel --delay 0.2 -j 8"     # -j N means N jobs at a time run in parallel
command+=" python run_compiling_cluster.py"
command+=" -n {1}"
command+=" -J0 {2}"
command+=" -J1 {3}"
command+=" -tt {4}"
command+=" -cbd {5}"
command+=" -l {6}"
command+=" -mi {7}"
command+=" -sc {8}"
command+=" -b {9}"
command+=" -om {10}"
command+=" -mpst {11}"
command+=" -mpsbd {12}"
command+=" -cd {13}"
command+=" -dpc {14}"
command+=" -ld {15}"

$command \
::: ${qubits[@]} \
::: ${even_coupling[@]} \
::: ${odd_coupling[@]} \
::: ${target_truncs[@]} \
::: ${compressed_bond_dims[@]} \
::: ${layers[@]} \
::: ${max_iter[@]} \
::: ${sufficient_cost[@]} \
::: ${backend[@]} \
::: ${optimiser_methods[@]} \
::: ${mps_truncation[@]} \
::: ${mps_bond_dim[@]} \
::: ${checkpoint_dir[@]} \
::: ${delete_prev_chkpt[@]} \
::: ${log_dir[@]}