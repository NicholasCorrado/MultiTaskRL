#!/bin/bash

pid=$1  # ranges from 0 to num_commands*num_jobs-1 
step=$2 # ranges from 0 to num_jobs-1
cmd=`tr '*' ' ' <<< $3` # replace * with space
echo $cmd $pid $step

git clone -b chtc https://github.com/NicholasCorrado/MultiTaskRL.git
cd MultiTaskRL
pip install -e custom-envs

# run your script -- $step ensures seeding is consistent across experiment batches
#python $cmd --run_id $step --seed $step
$($cmd --run_id $step --seed $step)

# compress results. This file will be transferred to your submit node upon job completion.
tar czvf results_${pid}.tar.gz results
mv results_${pid}.tar.gz ..
