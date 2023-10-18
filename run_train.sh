#!/bin/bash 
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate test310

# Loop 5 times
for i in {1..5}
do
   python3 --version
   python3 ./train.py
done


