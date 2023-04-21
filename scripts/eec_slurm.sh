#!/bin/bash

# name of job
# man 1 qsub
#$ -N eec

# use current working directory
#$ -cwd

#$ -pe parallel-onenode 64

# interpret using BASH shell
#$ -S /bin/bash

# join standard error and standard output of script into job_name.ojob_id
#$ -j y -o eec_output

# export environment variables to job
#$ -V

# when am I running
/bin/date

# where am I running
/bin/hostname

# what environment variables are available to this job script, e.g. $JOB_ID
/usr/bin/env

python eec_similarity.py -t 100
