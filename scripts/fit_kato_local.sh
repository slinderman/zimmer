#!/bin/sh

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "Fit the Kato dataset with D = " $1 " continuous dims"
cd ~/Projects/zimmer
parallel --results logs python experiments/fit_slds.py kato {} $1 --results_dir=results/kato/2018-11-20 --transitions=rbf --eta=1e-1 ::: 2 4 6 8 10 12 14 16 18 20 
echo all processes complete
python 
 
# End of script
