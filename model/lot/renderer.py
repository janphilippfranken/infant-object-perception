import torch
import itertools

from data import *
from return_types import *

def derender(a: DataObject) -> Bitmask:
        timesteps, h, w = a.codebooks.shape
        codes, counts = a.codebooks.unique(return_counts=True)
        background = codes[torch.argmax(counts)]
        codes = codes.masked_select(codes != background) # get rid of background
        permutations = list(itertools.permutations(range(codes.shape[0]))) # factorial
        n_codes, n_permutations = codes.shape[0], len(permutations)
        bitmasks = torch.zeros(timesteps, n_codes, h, w) 
        union = torch.zeros(timesteps, n_permutations, n_codes, h, w) 
        for i, c in enumerate(codes):
            bitmasks[:, i] = (a.codebooks[:, ] == c).int() 
        for i, c in enumerate(codes):
            for j, perm in enumerate(permutations):
                bitmask = bitmasks[:, perm]
                union[:, j, i] = torch.sum(bitmask[:, perm.index(i):], dim=1)
        return Bitmask(bitmasks, codes, permutations, union, background)

def render(a: Bitmask) -> Codebook:
    t, c, h, w = a.bitmasks.shape
    codebooks = torch.zeros(t, h, w)
    codebooks[torch.where(codebooks == 0)] = a.background.float()
    for i, c in enumerate(a.codes):
        codebooks[a.bitmasks[:, i, :, :].bool()] = c.float()
    return codebooks