import pandas as pd
import glob
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from tqdm import tqdm

EMOTION = "anger"

ARRAY_N_TASKS = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
ARRAY_TASK_NUM = int(os.environ["SLURM_ARRAY_TASK_ID"])


# reference embeddings
ref_embeddings = pd.read_parquet("embeddings/embeddings_e=reference_p=0_0.parquet")
ref_sents = pd.read_parquet("data/sentences/sentences_e=reference_p=0.parquet")
ref_embeddings = ref_embeddings.merge(ref_sents, on="sentence_index")
ref_embeddings = ref_embeddings[ref_embeddings['Emotion'] == EMOTION]

# sentence embeddings

# split df into number of tasks


fpaths = glob.glob(f"embeddings/embeddings_e={EMOTION}*.parquet")
fpaths = np.array_split(fpaths, ARRAY_N_TASKS)[ARRAY_TASK_NUM]

for fpath in tqdm(fpaths):
    sentence_embeddings = pd.read_parquet(fpath)

    # sim scores
    cls_sim_scores = cosine_similarity(np.vstack(sentence_embeddings['cls_embedding'].values), 
                                    np.vstack(ref_embeddings['cls_embedding'].values))
    mean_cls_sim_score = cls_sim_scores.mean(axis=1)


    mean_pool_sim_scores = cosine_similarity(np.vstack(sentence_embeddings['mean_embedding'].values),
                                            np.vstack(ref_embeddings['mean_embedding'].values))
    mean_mean_pool_sim_score = mean_pool_sim_scores.mean(axis=1)


    sentence_embeddings['mean_cls_sim_score'] = mean_cls_sim_score
    sentence_embeddings['mean_mean_pool_sim_score'] = mean_mean_pool_sim_score
    sentence_embeddings['cls_sim_scores'] = cls_sim_scores.tolist()
    sentence_embeddings['mean_pool_sim_scores'] = mean_pool_sim_scores.tolist()
    
    sentence_embeddings.to_parquet(fpath)