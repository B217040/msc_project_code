#!/bin/bash
print('stuff for the log')
with open('/work/tc046/tc046/pchamp/results/slurmtest.txt', 'a') as f:
    f.write('Testing slurm')