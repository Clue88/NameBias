"""
get embeddings for sentences
"""
import os
from transformers import AutoModel, AutoTokenizer, pipeline
from transformers.pipelines import FeatureExtractionPipeline
from transformers.pipelines.pt_utils import KeyDataset
import torch
import datasets
import pandas as pd
from tqdm.auto import tqdm
import numpy as np

# params
EMOTION = "anger"
MODEL_NAME = "roberta-base"
BATCH_SIZE = 128
JOB_NUM = int(os.environ["SLURM_ARRAY_TASK_ID"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, use_fast=True, return_tensors="pt"
)
model = AutoModel.from_pretrained(MODEL_NAME)

pipe = FeatureExtractionPipeline(
    model=model, tokenizer=tokenizer, device=0, return_tensors=True
)

for param in model.parameters():
    param.requires_grad = False


# -------------------------------------------------------------------------------------------

dataset = datasets.load_dataset(
    "parquet",
    data_files=[f"data/sentences/sentences_e={EMOTION}_p={JOB_NUM}.parquet"],
    columns=["name_index", "name", "Sentence", "sentence_index"],
    cache_dir="/scratch/pp1994/hfcache",
)["train"]

dataset = dataset.map(
    lambda batch: tokenizer(batch["Sentence"], truncation=True, padding=True),
    batched=True,
)

dataset.set_format(
    "torch", columns=["input_ids", "attention_mask", "name_index", "sentence_index"]
)


def collate_fn(examples):
    """
    collate function for dataloader
    """
    return tokenizer.pad(examples, padding="longest", return_tensors="pt").to(device)


dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False
)


def mean_pooling(model_output, attention_mask):
    """
    mean pooling of tokens, taking into account attention mask
    """
    # see https://www.sbert.net/examples/applications/computing-embeddings/README.html#sentence-embeddings-with-transformers
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = (attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def create_write_buffer():
    """
    create write buffer
    """
    return {
        "cls_embedding": [],
        "mean_embedding": [],
        "name_idx": [],
        "sentence_idx": [],
    }

def create_df_from_buffer(write_buffer):
    """
    create dataframe from write buffer
    """
    df = pd.DataFrame(
        {
            "cls_embedding": np.concatenate(write_buffer["cls_embedding"]).tolist(),
            "mean_embedding": np.concatenate(write_buffer["mean_embedding"]).tolist(),
            "name_idx": np.concatenate(write_buffer["name_idx"]),
            "sentence_idx": np.concatenate(write_buffer["sentence_idx"]),
        }
    )
    return df

write_buffer = create_write_buffer()
part_num = 0

for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    # output from transformer model
    # output[0] corresponds to hidden layers of all tokens
    # output[1] corresponds to pooler layer equiv to output.pooler_output
    # pooler_output takes [CLS] for each sequence then applies linear and tanh layer
    # output[0] has dimension (batch_size, max_num_tokens, hidden_size)
    # output[0][:, 0, :] corresponds to the first token <s> equivalent to [CLS]
    # for pooler_output vs [CLS] token see https://github.com/huggingface/transformers/issues/7540
    cls_embeddings = out[0][:, 0, :].detach().cpu().numpy()
    mean_embeddings = mean_pooling(out, batch["attention_mask"]).detach().cpu().numpy()
    write_buffer["cls_embedding"].append(cls_embeddings)
    write_buffer["mean_embedding"].append(mean_embeddings)
    write_buffer["name_idx"].append(batch["name_index"].detach().cpu().numpy())
    write_buffer["sentence_idx"].append(batch["sentence_index"].detach().cpu().numpy())
    if (i % 100 == 0) & (i > 0):
        # write to file every 100 batches (128k sentences)
        df = create_df_from_buffer(write_buffer)
        df.to_parquet(
            f"embeddings/embeddings_e={EMOTION}_p={JOB_NUM}_{part_num}.parquet"
        )
        write_buffer = create_write_buffer()
        part_num += 1


df = create_df_from_buffer(write_buffer)
df.to_parquet(f"embeddings/embeddings_e={EMOTION}_p={JOB_NUM}_{part_num}.parquet")
