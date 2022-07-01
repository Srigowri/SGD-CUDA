#!/usr/bin/env bash

mkdir -p results
commit=`git rev-parse --short HEAD`
date=`date '+%Y-%m-%d-%H-%M-%S'`
filename="results/$date-$commit.txt"
datasets=('ml-1m' 'ml-10M100K')
iterations=(100 500 1000 5000 10000)
factors=(50 300)

for dataset in "${datasets[@]}"; do
    for iteration in "${iterations[@]}"; do
        for factor in "${factors[@]}"; do
            python3 ../preprocessing/create_config.py exp.cfg -n "$iteration" -f "$factor"
            { time ../a.out -c exp.cfg "../data/$dataset/ratings_mapped_train.dat" "../data/$dataset/ratings_mapped_test.dat" ; } >> $filename 2>&1
            echo "Done with $factor factors with $iteration iterations on $dataset" | tee -a $filename
        done
    done
done


datasets=('ml-20m' 'ml-25m')
for dataset in "${datasets[@]}"; do
    for iteration in "${iterations[@]}"; do
        for factor in "${factors[@]}"; do
            python ../preprocessing/create_config.py exp.cfg -n "$iteration" -f "$factor"
            { time ../a.out -c exp.cfg "../data/$dataset/ratings_mapped_train.csv" "../data/$dataset/ratings_mapped_test.csv" ; } >> $filename 2>&1
            echo "Done with $factor factors with $iteration iterations on $dataset" | tee -a $filename
        done
    done
done

