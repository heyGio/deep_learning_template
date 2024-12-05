#!/bin/bash

experiment_name="batch_size"

batch_sizes=(16 64 128)

for b in "${batch_sizes[@]}";
do
    echo "Running experiment $experiment_name with batch_size=$b"
    python train.py --experiment_name $experiment_name --run_name $experiment_name$b --batch_size $b
done
