#!/bin/bash

nsteps=4000000000

#temps=1.0
temps=1.0

dt=0.005
gamma=0.001
mass=1.0
seed=1


#    parser.add_argument("--lattice_repeats",default=5,type=int,help="times to repliacte the system in each direction (default: %(default)s)") #no of times we want to replicate in one direction
#    parser.add_argument("--lattice_type",default="bcc",help="Lattice type (bcc, sc, fcc) (default: %(default)s)")

submit_simulation () {
    repeats=$1
    lattice_spacing=$2
    radiusP=$3
    radiusN=$4
    fraction_positive=$5
    debye=$6
    brush_length=$7
    prev_steps=$8
    otheroptions=""
 
    if [ ! -z "$prev_steps" ];then
        final_steps=$(($prev_steps+$nsteps))
    else
        final_steps=$nsteps
    fi 

    prefix=prod2_v2/bulk_LS${lattice_spacing}_FP${fraction_positive}_RN${radiusN}_BL${brush_length}_DL${debye}/trajectory_LS${lattice_spacing}_FP${fraction_positive}_RN${radiusN}_BL${brush_length}_DL${debye}
    outprefix=${prefix}_N${final_steps}
    input_file=${prefix}_N${prev_steps}.gsd
    if [ -e "$input_file" -o -e "${outprefix}.gsd" ];then
        if [ ! -z "$prev_steps" ];then
            jobname=$(basename $outprefix)
            jobdir=$(dirname $outprefix)
            simulation_options="--gpu --radiusN $radiusN --radiusP $radiusP -i $input_file -n $nsteps -d $debye --dt $dt --gamma $gamma -T $temps --seed -B $brush_length $seed $otheroptions" 
            sbatch --job-name=$jobname -o ${outprefix}.slurm_%j.log -o ${outprefix}.slurm_%j.log --export=outprefix=$outprefix,simulation_options="${simulation_options}" run_simulation_v2.sbatch
        fi
    else
        jobname=$(basename $outprefix)
        jobdir=$(dirname $outprefix)
        mkdir -p $jobdir
        simulation_options="--lattice_repeats $repeats --radiusN $radiusN --radiusP $radiusP --gpu -n $nsteps -d $debye --lattice_spacing $lattice_spacing $packing_fraction --dt $dt --gamma $gamma -T $temps --fraction_positive $fraction_positive -B $brush_length --seed $seed $otheroptions" 
        sbatch --job-name=$jobname -o ${outprefix}.slurm_%j.log --export=ALL,outprefix=$outprefix,simulation_options="${simulation_options}" run_simulation_v2.sbatch
#run_simulation_v2_minfirst.sbatch
    fi
}
#just test 1

RP=200
repeats=15
#submit_simulation $repeats 2.5 $RP 100 0.1 6
#exit

#P is fist, N is second
#fraction is fraction positive
for debye in 5.45 5.5 5.55 5.6 5.65;do
    #for lattice_spacing in 2.3 2.5 2.7;do
    #for lattice_spacing in 2.1 2.0;do
    for lattice_spacing in 4 4.5 5;do # 4.5 5 5.5 6;do
        for RN in 185;do
            for frac_positive in 0.5;do
                for brush_length in 10;do
                    submit_simulation $repeats $lattice_spacing $RP $RN $frac_positive $debye $brush_length
                done
           done
        done
    done
done
