from flair.embeddings import BertEmbeddings
from flair.data import Sentence
import numpy as np
from numpy.linalg import norm
import pandas as pd
import scipy
import torch


EMBEDDING = BertEmbeddings("bert-large-uncased")
EEC = pd.read_csv("Equity-Evaluation-Corpus.csv", header=0)


def sent_emb(sent):
    """
    Embeds a given sentence using BertEmbeddings.
    """
    sent = Sentence(sent)
    EMBEDDING.embed(sent)
    se = torch.stack([token.embedding for token in sent])
    return np.average(se.cpu().numpy(), axis=0)


def cos_sim_raw_sent_emb(ref_sen_emb, sent1, sent2):
    """
    Finds the cosine similarity between two sentence embeddings and a
    reference embedding.
    """
    # use Flair built-in Sentence classes
    sentence1 = Sentence(sent1)
    sentence2 = Sentence(sent2)

    EMBEDDING.embed(sentence1)
    se1 = torch.stack([token.embedding for token in sentence1])
    sent1_emb = se1.cpu().numpy()

    EMBEDDING.embed(sentence2)
    se2 = torch.stack([token.embedding for token in sentence2])
    sent2_emb = se2.cpu().numpy()

    # get average sentence embeddings
    sent1_emb = np.average(sent1_emb, axis=0)
    sent2_emb = np.average(sent2_emb, axis=0)

    # cosine similarity
    proj1 = (sent1_emb / norm(sent1_emb)).dot(ref_sen_emb) / norm(ref_sen_emb)
    proj2 = (sent2_emb / norm(sent2_emb)).dot(ref_sen_emb) / norm(ref_sen_emb)

    return proj1, proj2


def embed_name(name, gender_identity, emotion):
    """
    Embeds a name with a given gender identity and emotion.
    """
    if gender_identity == "male":
        sentences = (
            EEC[(EEC.Person == "Alonzo") & (EEC.Emotion == emotion)]
            .Sentence.str.replace("Alonzo", name)
            .tolist()
        )
    elif gender_identity == "female":
        sentences = (
            EEC[(EEC.Person == "Katie") & (EEC.Emotion == emotion)]
            .Sentence.str.replace("Katie", name)
            .tolist()
        )
    return sentences


def compare_names(name1, sex1, name2, sex2, emotion, ref):
    n1 = embed_name(name1, sex1, emotion)
    n2 = embed_name(name2, sex2, emotion)

    reference_sent = sent_emb(ref)

    n1_sim = []
    n2_sim = []
    for a, b in zip(n1, n2):
        s1, s2 = cos_sim_raw_sent_emb(reference_sent, a, b)
        n1_sim.append(s1)
        n2_sim.append(s2)

    p1 = scipy.stats.ttest_rel(n1_sim, n2_sim)
    print(
        np.average(n1_sim),
        np.average(n2_sim),
        np.average(n1_sim) - np.average(n2_sim),
        p1.statistic,
        p1.pvalue,
    )


if __name__ == "__main__":
    compare_names("Candido", "male", "Athanasios", "male", "anger", "I feel angry")

# df[df["is_first_name"] == "T"][(df["pred_race"] == "nh_white") | (df["pred_race"] == "nh_black")][df["frequency_male"] > 10][df["frequency_pile"] > 14000][df["frequency_pile"] < 16000][df["num_tokens"] == 4]
