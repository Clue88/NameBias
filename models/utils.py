import torch


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