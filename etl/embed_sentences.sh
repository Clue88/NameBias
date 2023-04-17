#!/bin/bash

# name of job
#$ -N embed_sentences_sadness

# working directory
#$ -wd /nlp/data/liuchris/NameBias/

#$ -pe parallel-onenode 4

# join standard error and standard output of script into job_name.ojob_id
#$ -j y -o embed_sentences_sadness.out

env/bin/python3 etl/embed_sentences.py
