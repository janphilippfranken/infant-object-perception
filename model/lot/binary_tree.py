class BinaryTree():
    """
    Generic binary tree

    """
    def __init__(self, 
                 val=None, 
                 left=None, 
                 right=None,
                ):
        self.val = val
        self.left = left
        self.right = right
      
    def add_node(self, 
                 val,
                ):  
        if self.val:
            if self.val > val:
                if self.left is None:
                    self.left = BinaryTree(val)
                else:
                    self.left.add_node(val)
            elif self.val < val:
                if self.right is None:
                    self.right = BinaryTree(val)
                else:
                    self.right.add_node(val) 
        else:
            self.val = val

    def post_order_traversal(self, 
                             root,
                            ):
        res = []
        if root:
            res = self.post_order_traversal(root.left)
            res = res + self.post_order_traversal(root.right)
            res.append(root.val)
        return res

    def in_order_traversal(self, 
                             root,
                            ):
        res = []
        if root:
            res = self.in_order_traversal(root.left)
            res.append(root.val)
            res = res + self.in_order_traversal(root.right)
        return res

    def get_non_terminals(self,
                          root,
                          ):
        res = []
        if root:
            res = self.get_non_terminals(root.left)
            res = res + self.get_non_terminals(root.right)
            if root.left or root.right:
                res.append(root.val)
        return res