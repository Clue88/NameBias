#!/bin/bash
#SBATCH --job-name=embed_sentences_sadness
#SBATCH --output=embed_sentences_sadness.out
#SBATCH --partition=p_nlp
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00
#SBATCH --mem=1000GB
#SBATCH -D /nlp/data/liuchris/NameBias/

python3 etl/embed_sentences.py
