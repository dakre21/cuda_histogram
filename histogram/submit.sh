```bash 
#!/bin/bash 
#BSUB -n 32 
#BSUB -o histogram.out 
#BSUB -e histogram.err 
#BSUB -q "windfall" 
#BSUB -J histogram
#BSUB -R gpu 
#--------------------------------------------------------------------- 

if [[ -z ${input} || -z ${output} ]]; then
    echo "Invalid input arguments"
    echo "For example 'bsub -a input=<file_name> output=<file_name>'"
    exit 0
fi

module load cuda  
cd /extra/dakre/cuda_histogram/histogram
time ./histogram ${input} ${output} 
###end of script 
``` 
