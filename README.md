# Intracranial Pulsation

This repository contains the code, meshes and data required to reproduce the results of the article "Human intracranial pulsatility during the cardiac cycle: a computational modelling framework", M. Causemann, V. Vinje and M.E. Rognes (2022).

The easiest way of getting the code running is based on snakemake and singularity: Installing both (using .e.g conda:  conda create -c bioconda -c conda-forge -n snakemake snakemake singularity) and running snakemake --cores N will run pull a docker image (mcause/brainsim:openblas) and execute all steps on N cores. Note however, that the required computational resources are quite large and unlikely to be met by a desktop computer. Snakemake also supports submitting jobs on a SLURM cluster via the --profile profile_folder_name option. An example of a profile configuration can be found in the ex3 folder.
The complete workflow is defined in the  Snakefile , which hence also provides a documentation of the indivudual steps.