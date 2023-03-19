import torch
import numpy as np

from primitive import Primitive
from return_types import *



class Reducer(Primitive): 

    def __init__(self):
        super().__init__(*Reducer.__dict__.values())

    def _bitmask(self, a: Bitmask) -> Bitmask: 
        return Bitmask(a.bitmasks)

    
class BitmaskFunction(Primitive):

    def __init__(self):
        super().__init__(*BitmaskFunction.__dict__.values())
    
    def _complement(self, a: Bitmask) -> Bitmask:
        return Bitmask((a.bitmasks != 1).int())
    
    def _constant(self, a: Bitmask) -> Bitmask:
        return Bitmask(a.bitmasks)
    
    def _move_x(self, a: Bitmask, n: int) -> Bitmask:
        return Bitmask(torch.roll(a.bitmasks, n))
    
    def _move_y(self, a: Bitmask, n: int) -> Bitmask:
        return Bitmask(torch.roll(a.bitmasks, n, dims=1))
   

class NumberFunction(Primitive):

    def __init__(self):
        super().__init__(*NumberFunction.__dict__.values())
    
    def _modulus(self, a: Number, n: int) -> Number:
        return Number(int(np.round(torch.remainder(a.n, n))))

    def _multiply(self, a: Number, n: int) -> Number:
        return Number(int(np.round(a.n * n)))

    def _neg(self, a: Number, n: int=1) -> Number:
        return Number(int(-a.n))

    def _add(self, a: Number, n: int) -> Number:
        return Number(int(np.round(a.n + n)))

    def _subtract(self, a: Number, n: int) -> Number:
        return Number(int(np.round(a.n - n)))

    def _constant_n(self, a: Number, n: int) -> Number:
        return Number(int(np.round(n)))

    def _division(self, a: Number, n: int) -> Number:
        return Number(int(np.round(a.n / n)))


class SetFunction(Primitive):

    def __init__(self):
        super().__init__(*SetFunction.__dict__.values())

    def union(self, a: Bitmask, b: Bitmask) -> Bitmask:
        return Bitmask(torch.logical_or(a.bitmasks, b.bitmasks).int())

    def intersection(self, a: Bitmask, b: Bitmask) -> Bitmask:
        return Bitmask(torch.logical_and(a.bitmasks, b.bitmasks).int())