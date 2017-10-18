#!/bin/bash 
#BSUB -n 16
#BSUB -o histogram.out 
#BSUB -e histogram.err 
#BSUB -q "windfall" 
#BSUB -J histogram
#BSUB -R gpu 

export IN_FILE="input_example.txt"
export OUT_FILE="output_result.txt"

module load cuda  
cd /extra/dakre/cuda_histogram/histogram
time ./histogram ${IN_FILE} ${OUT_FILE}
###end of script 
