import datasets
from transformers import AutoTokenizer
import pandas as pd


MODEL_NAME = "roberta-base"

dataset = datasets.load_dataset("parquet",
                                    data_files=[f"data/names.parquet"],
                                    cache_dir="/scratch/pp1994/hfcache",)["train"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, return_tensors=False)

def get_n_tokens(batch):
    batch['n_tokens'] = tokenizer(batch['name'], truncation=False, padding=False, return_length=True)['length']
    return batch

dataset = dataset.map(get_n_tokens, batched=True)

df = dataset.to_pandas()
df.to_parquet("data/names.parquet")

df = pd.read_parquet("data/names.parquet")