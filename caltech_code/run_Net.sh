#!/bin/bash

#SBATCH -J caltech1
#SBATCH -o out/out.%j.out
#SBATCH -N 1
#SBATCH -n 16

g++ -lm -O2 -Wall -std=c++17 *.cpp -o main

./main