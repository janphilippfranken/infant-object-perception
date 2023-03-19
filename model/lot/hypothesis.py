import torch
import numpy as np
import copy

from scipy.stats import geom


from return_types import *
from grammar import *
from binary_tree import *
from number import *

# instantiate primitives
R = Reducer()
B = BitmaskFunction()
S = SetFunction()


def number_function(t, 
                    p=0.4, 
                    n: int=5,
                    ):
    """
    run nested number transformation
    """
    time_H = Node(torch.tensor(1))
    n_expansions = geom.rvs(p=p, size=1)[0]
    nodes = torch.arange(2, 2 + n_expansions)
    for n in nodes:
        time_H._add_node(n)
    time_H._traverse(time_H, t=t)
    return time_H, torch.log(One / n_expansions)


class HypothesisTree(BinaryTree):
    """
    Main hypothesis class running on bitmask (binary tree data structure)
    
    """
    def __init__(self, 
                 val=None, 
                 left=None, 
                 right=None,
                 _return=None,
                  n:int=5,
                ):
        super().__init__(BinaryTree)
        self.val = val
        self.left = left
        self.right = right
        self._return = _return
        self.n = n
        self._methods = [] 
        self.reducer = None
        self.prior = 0
        self.likelilhood = 0
    
    def _traverse(self, 
                  root,
                  data,
                  t: int=0,
                  ):
        """Build tree using postorder traversal
        
          Args:
           
          Returns:
        """
        returns = []
        
        if root:
            returns = self._traverse(root.left, data) # postorder traversal
            returns = returns + self._traverse(root.right, data)
            returns.append(root.val)
            
            if not root.left and not root.right: # terminal
                self.reducer = R._inputs[Bitmask][0]
                reduced = self.reducer(None, data)
                root._return = reduced
                self._methods.append(self.reducer)
                
            elif root.left and root.right: # merge 
                methods = S._inputs[type(root.left._return)]
                method = np.random.choice(methods)
                res = method(None, root.left._return, root.right._return)
                self._methods.append((method))
                root._return = res
               
            elif root.left and not root.right: # left 
                methods = B._inputs[type(root.left._return)]
                method = np.random.choice(methods)
                if "n" in method.__annotations__.keys(): # method requires n eg (move x by n)
                    time_h, prior_node_prob = number_function(Number(t))
                    res = method(None, root.left._return, 0)
                    self._methods.append((method, time_h))
                else:
                    res = method(None, root.left._return)
                    self._methods.append((method))
                root._return = res
                
            elif root.right and not root.left: # right
                methods = B._inputs[type(root.right._return)]
                method = np.random.choice(methods)
                if "n" in method.__annotations__.keys(): # method requires n eg (move x by n)
                    time_h, prior_node_prob = number_function(Number(t))
                    res = method(None, root.right._return, 0)
                    self._methods.append((method, time_h))
                else:
                    res = method(None, root.right._return)
                    self._methods.append((method))
                root._return = res
                
        return returns

    def _evaluate(self,
                  root,
                  data,
                  methods,
                  t,
                 ):
        """Test hypothesis against data 
        
          Args:
           
          Returns:
        """
        returns = []

        if root:
            
            returns = self._evaluate(root.left, data, methods, t) # postorder traversal
            returns = returns + self._evaluate(root.right, data, methods, t)
            returns.append(root.val)

            if not root.left and not root.right: # terminal
                method = methods.pop(0)
                bitmask = method(None, data)
                root._return = data
                returns.append(bitmask)

            elif root.left and root.right: # merge
                method = methods.pop(0) 
                res = method(None, root.left._return, root.right._return)
                root._return = res
                returns.append(res)

            elif root.left and not root.right: # left 
                method = methods.pop(0) 
                try:
                    n = method[1]._evaluate(method[1], t=Number(t), methods=copy.deepcopy(method[1]._methods))[-1].n
                    res = method[0](None, root.left._return, n) # bitmask operation
                except:
                    res = method(None, root.left._return)
                root._return = res
                returns.append(res)

            elif root.right and not root.left: # right
                method = methods.pop(0) 
                try:
                    n = method[1]._evaluate(method[1], t=Number(t), methods=copy.deepcopy(method[1]._methods))[-1].n
                    res = method[0](None, root.right._return, n) # bitmask operation
                except:
                    res = method(None, root.right._return)
                root._return = res
                returns.append(res)

        return returns