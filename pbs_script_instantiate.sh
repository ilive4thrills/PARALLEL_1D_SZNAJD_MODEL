# Author: Joseph H. Garcia
# Date: January 30th, 2017
# Filename: pbs_script_instantiate.sh
# Purpose: This script takes the below template (TEMPLATE.TXT) and creates many copies of it, each with a different 
# combination of the parameters L (SYS_SIZES) and p (p_vals) substituted in for SYS_SIZE and PVAL. Each one of those
# copied files is taken by the Bash script send_to_pbs.sh and put in the PBSjob queue. 

# BEGIN CONTENTS OF TEMPLATE.TXT #
###########################################
##!/bin/bash -l
##PBS -q batch
##PBS -N PBS_SYS_SIZE_PVAL_NOR_NUMPROCS_p
##PBS -l procs=NUMPROCS
##PBS -l walltime=340:00:00

##PBS -o NSYS_SIZE_PVAL_NOR_NUMPROCS_p.txt
##PBS -e NSYS_SIZE_PVAL_NOR_NUMPROCS_pe.txt

#cd $PBS_O_WORKDIR

#module load openmpi/gnu/1.8.1

#mpirun -np NUMPROCS ./MPI1_2_XRANGE_ONE_UPDATE SYS_SIZE PVAL NOR BEGIN_X END_X  # have to make sure my C-code knows which argument is which
##########################################
# END CONTENTS OF TEMPLATE.TXT #

SYS_SIZES=(100 500 1000 5000 10000 50000 100000)
p_vals=(0.000 0.001 0.0025 0.005 0.0075 0.01 0.025 0.050 0.075 0.1 0.25 0.50 0.75 1.0)
NOR=5000    # do 5000 iterations at each x-value 
NUMPROCS=10   # have 10 processors (nodes, cores, etc.) do the work

for i in {0..6..1}
do 
	for j in {0..13..1}
	do	
		touchstring=$(echo "$PWD/PBS_SCRIPTS/PBS_${SYS_SIZES[i]}_${p_vals[j]}_${NOR}_${NUMPROCS}p"	)     # print the batch job file names here
		touch $touchstring                                                                 # Create the batch job file
		echo $touchstring                                                                  # Print the name of that batch job file
		sed -e "s/SYS_SIZE/${SYS_SIZES[i]}/" -e "s/PVAL/${p_vals[j]}/" -e "s/NOR/${NOR}/" -e "s/NUMPROCS/${NUMPROCS}/" template.txt > $touchstring      # modify the DUMMY variable names in the template-based file
		# make sure that C code is compiled and that that executable is ssvbbbb
	done
done


