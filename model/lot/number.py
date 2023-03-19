import torch
import numpy as np

from return_types import *
from grammar import *


# number function primitive
N = NumberFunction()


class Node():
    """
    Number component of hypothesis
    
    """
    def __init__(self, 
                 node, 
                 _next=None, 
                 _return=None,
                 n:int=5,
                ):
        self.node = node 
        self.next = _next
        self._return = _return
        self.n = n
        self._methods = [] 
        self.reducer = None
        
    def _add_node(self,
                  node):
        if self.node:  
            if self.next is None:
                self.next = Node(node)
            else:
                self.next._add_node(node)
        else:
            self.node = node
             
    def _traverse(self,
                  root,
                  t: Number,
                 ):
        returns = []
        if root:
            returns = self._traverse(root.next, t)
            returns.append(root.node)
            if self.reducer is None: # terminal
                self.reducer = Number
                reduced = t
                root._return = reduced
                self._methods.append(self.reducer) 
            else:                    # non terminal
                methods = N._inputs[type(root.next._return)]
                method = np.random.choice(methods)             
                n = torch.tensor(np.random.randint(1, self.n + 1))        
                res = method(None, root.next._return, n)
                self._methods.append((method, n))
                root._return = res
        return returns 
    
    def _evaluate(self,
                  root,
                  methods,
                  t,
                 ):
        returns = []
        if root:
            returns = self._evaluate(root.next, methods, t) 
            if root.next is None: # terminal
                method = methods.pop(0)
                root._return = t
                returns.append(t)
            else:                 # non terminal  
                method = methods.pop(0) 
                res = method[0](None, root.next._return, method[1]) 
                root._return = res
                returns.append(res)
        return returns