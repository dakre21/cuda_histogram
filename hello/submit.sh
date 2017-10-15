```bash 
#!/bin/bash 
#BSUB -n 32 
#BSUB -o hello-world.out 
#BSUB -e hello-world.err 
#BSUB -q "windfall" 
#BSUB -J hello_world 
#BSUB -R gpu 
#--------------------------------------------------------------------- 

module load cuda  
cd /extra/dakre/cuda_histogram/hello
time ./hello
###end of script 
``` 
