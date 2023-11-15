# -, *, +, *, x, *, /, x, x, 2.0, x, 6.0, 1.5, 1.2, x
# f(x) = x^3 - 2x + 4

from metaheuristic_designer import Encoding

class GPEEncoding(Encoding):
    def __init__(self, max_size, op_size, operators=None):
        self.max_size=max_size
        self.op_size=op_size
        if operators is None:
            operators=[("+", 2), ("-", 2), ("*", 2), ("/", 2)]
        self.operators = operators

    def encode(self, phenotype):
        """
        Too hard to implement and it's not going to be used anyway. :V
        """

        return np.zeros(self.max_size)
    
    # def _construct_tree(genotype, cursor_parent=0, cursor_child=0):
    #     t = 0
    #     if cursor_parent < self.op_size:
    #         op_code = genotype[cursor_parent]
    #         op_str, op_ar = self.operators[op_code]

    #         subtrees = []
    #         for i in range(op_ar):
    #             #new_tree, cursor_child = construct_tree(genotype, cursor_child+i, cursor_child)
    #             construct_tree(genotype, t, t)
    #             subtrees.append(new_tree)
    
    #         expr_tree = [op_str, subtrees]

    #         return expr_tree, cursor_child
    #     else:
    #         return 0, 0
    
    def _construct_tree(genotype, cursor_parent=0, cursor_child=0):
        """
        BFS traversal
        """

        
    
    def decode(self, genotype):
        """
        """

        expr_tree = self._construct_tree(genotype)



        


