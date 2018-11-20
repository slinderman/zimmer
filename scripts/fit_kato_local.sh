#!/bin/sh

echo "Fit the Kato dataset with D = " $1 " continuous dims"
cd ~/Projects/zimmer

parallel experiments/fit_slds.py kato {} $1 --results_dir=results/kato/2018-11-20 --transitions=rbf ::: 2 4 6 8 10 12 14 16 18 20
echo all processes complete
python 
 
# End of script