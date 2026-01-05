#!/bin/bash

#-------------------------------------------------
# Bash script to run GMF over multiple geometries
#-------------------------------------------------

cwd=$(pwd)

cd geom_00 
echo "Starting" 
python ../gmf_csf.py > log.txt 
cd .. 

prev=00
for i in {01..20}; do 
    echo "Step $i"
    cd geom_$i
    python ../gmf_csf.py ../geom_$prev/curr_wfn > log.txt  
    cd .. 
    prev=$i
done

cd $cwd

