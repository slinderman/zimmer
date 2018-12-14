#!/bin/sh

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd ~/Projects/zimmer
parallel --no-notice --results logs python experiments/fit_arhmm.py kato {} --no_hierarchical --results_dir=results/kato/2018-12-11 --transitions=recurrent --eta=1e-1 ::: 2 4 6 8 10 12 14 16 18 20 
parallel  --no-notice --results logs python experiments/fit_arhmm.py kato {} --results_dir=results/kato/2018-12-11 --transitions=recurrent --eta=1e-1 ::: 2 4 6 8 10 12 14 16 18 20 
echo all processes complete
 
# End of script
