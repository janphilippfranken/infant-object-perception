from typing import TypeVar
import torch

Tensor = TypeVar(torch.tensor)
One = torch.tensor(1)


class Codebook():
    def __init__(self, a: Tensor): 
        self.codebooks = a
        
class Bitmask():
    def __init__(self, bitmasks: Tensor, codes: Tensor=One, permutations: Tensor=One, union: Tensor=One, background: Tensor=One):
        self.bitmasks = bitmasks
        self.codes = codes
        self.permutations = permutations
        self.union = union
        self.background = background
        
class Number():
    def __init__(self, a: Tensor):
        self.n = a

class CodebookPair():
    def __init__(self, a1: Codebook, a2: Codebook):
        self.pair = torch.stack((a1.codebook, a2.codebook))
    
class BitmaskPair():
    def __init__(self, a1: Bitmask, a2: Bitmask):
        self.pair = torch.stack((a1.bitmask, a2.bitmask)) 

class NumberPair():
    def __init__(self, a1: Number,  a2: Number):
        self.pair = torch.stack((a1.n, a2.n))