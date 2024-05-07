#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mem 4GB
#SBATCH -t 48:00:00
#SBATCH --dependency=singleton
#module load hoomd-blue/openmpi/intel/2.6.1

module purge
module load python/intel/3.8.6
module load intel/19.1.2
module load openmpi/intel/4.0.5
module load cuda/11.1.74
module load hoomd/cuda/openmpi/intel/2.9.3

module list


if [ -z "$outprefix" ];then
    echo "Output prefix required"
    exit
fi
if [ -e "${outprefix}.gsd" ];then
    echo "Skipping $outprefix, already ran"
    exit
fi

echo "from batch file" $outprefix
echo ${simulation_options}


outdir=$(dirname $outprefix)
mkdir -p $outdir
exe=dlvo_bulk.py

python $exe -o $outprefix ${simulation_options} > ${outprefix}.hoomd.log
