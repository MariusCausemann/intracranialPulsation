#!/bin/bash
# properties = {properties}
module purge
module load singularity-ce
module load mpich-3.3.2
#module load OpenBLAS/0.3.7-GCC-8.3.0
export MUMPS_OOC_TMPDIR=/tmp/
export SINGULARITYENV_PYVISTA_OFF_SCREEN=true
export SINGULARITYENV_YVISTA_USE_PANEL=false
export OMP_PLACES=cores
export OMP_PROC_BIND=close
#export SINGULARITYENV_OPENBLAS_NUM_THREADS=8
#export SINGULARITYENV_OMP_NUM_THREADS=12
#export OMP_NUM_THREADS=8
#export BLIS_NUM_THREADS=8
#unset MKL_NUM_THREADS
#unset SINGULARITYENV_MKL_NUM_THREADS
#export MKL_DYNAMIC=false
#export SINGULARITYENV_MKL_DYNAMIC=false
#export SINGULARITYENV_OMP_nested=true
#export SINGULARITYENV_OMP_DYNAMIC=false
#export SINGULARITYENV_MKL_DYNAMIC=false
#export SINGULARITYENV_MKL_NUM_THREADS=12
#export MKL_NUM_THREADS=18
#export OPENBLAS_NUM_THREADS=8
#export OMP_PLACES=threads 
#export OMP_PROC_BIND=spread
{exec_job}
