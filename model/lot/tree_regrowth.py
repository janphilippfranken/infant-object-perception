import numpy as np
import torch
import random
import random
import copy

from scipy.stats import geom

from return_types import *
from hypothesis import HypothesisTree


class TreeRegrowth():
    """
    tr style regrowth as proposal distribution (Goodman+ 2008 Cognitive Science)

    """
    def __init__(self, 
                 height: int=64, 
                 width: int=64, 
                 p: int=0.4,
                 ):
        self.p = p
        self.height = height
        self.width = width
        self.data = torch.zeros(self.height, self.width)

    def propose(self, 
                h,
                ):
        nt = h.get_non_terminals(h)
        rand_nt = np.random.choice(nt)
        rand_nt_idx = h.post_order_traversal(h).index(rand_nt)
        h_prime = HypothesisTree(1)
        values = np.arange(1002) # hack
        values = np.delete(values, h.post_order_traversal(h))
        size = geom.rvs(p=self.p, size=1)[0]
        nodes =  np.random.choice(values, size=size, replace=False)
        for n in nodes:
            h_prime.add_node(n)
        h_prime._traverse(h_prime, data=Bitmask(self.data))
        return h_prime, rand_nt, rand_nt_idx

    def replace_duplicates(self, 
                           root, 
                           exclude,
                           ):
        if root is None:
            return None
        values = set()
        self.replace_duplicates_helper(root, values, exclude)
        return root

    def replace_duplicates_helper(self, 
                                  node, 
                                  values, 
                                  exclude,
                                  ):
        if node is None:
            return
        if node.val in values:
            possible_values = set(range(1, 1001)) - set(exclude)
            node.val = random.choice(list(possible_values))
        else:
            values.add(node.val)
        self.replace_duplicates_helper(node.right, values, exclude)
        self.replace_duplicates_helper(node.left, values, exclude)

    def merge_trees(self, 
                    h, 
                    h_prime, 
                    key,
                    ):
        if h is None:
            return None
        if h.val == key:
            return h_prime
        else:
            h.left = self.merge_trees(h.left, h_prime, key)
            h.right = self.merge_trees(h.right, h_prime, key)
        return h

    def flatten_list(self, 
                     lst,
                     ):
        flattened_list = []
        for i in lst:
            if isinstance(i, list):
                flattened_list.extend(self.flatten_list(i))
            else:
                flattened_list.append(i)
        return flattened_list

    def regrowth(self, 
                 h,
                 ):
        h_prime, rand_nt, rand_nt_idx = self.propose(h)
        merge_tree = self.merge_trees(copy.deepcopy(h), copy.deepcopy(h_prime), rand_nt)
        merge_tree = self.replace_duplicates(merge_tree, exclude=h.post_order_traversal(h))
        nodes_h = set(h.post_order_traversal(h))
        nodes_merge_tree = set(merge_tree.post_order_traversal(merge_tree))
        common_nodes = nodes_h.intersection(nodes_merge_tree)
        merge_tree._methods = []
        for i, val in enumerate(h.post_order_traversal(h)):
            if val in common_nodes:
                merge_tree._methods.append(h._methods[i])
            else:
                merge_tree._methods.append(None)
        merge_tree._methods[rand_nt_idx] = copy.deepcopy(h_prime._methods)
        merge_tree._methods = self.flatten_list(merge_tree._methods)
        merge_tree._methods = list(filter(None, merge_tree._methods))
        return merge_tree, h_prime, rand_nt, rand_nt_idx