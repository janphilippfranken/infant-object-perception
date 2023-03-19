import torch

def _get_code_distribution(codes, n_embeddings):
    code_distribution = torch.zeros(n_embeddings).long()
    bins = torch.sort(torch.bincount(codes.flatten()))
    bins = torch.flip(bins.values, dims=[0])
    index = torch.arange(bins.shape[0])
    code_distribution.index_add_(0, index, bins)
    return code_distribution / torch.sum(code_distribution)