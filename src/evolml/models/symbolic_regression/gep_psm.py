from metaheuristic_designer import Encoding, ObjectiveVectorFunc
import numpy as np


class GEPEncoding(Encoding):
    """
    Gene expression programming encoding
    """

    def __init__(self, max_size, op_size, operators=None):
        self.max_size = max_size
        self.op_size = op_size
        if operators is None:
            operators = [("+", 2), ("-", 2), ("*", 2), ("/", 2), ("x", 0)]
        self.operators = operators

    def encode(self, phenotype):
        """
        Too hard to implement and it's not going to be used anyway. :V
        """

        return np.zeros(self.max_size)

    def _construct_tree(self, genotype):
        """
        Transform a vector into an expression tree.
        """

        # Go through the vector as a tree in BFS order
        op_code = int(genotype[0])
        op_node = self.operators[op_code]
        expr_tree = [op_node]
        node_queue = [expr_tree]

        cursor = 0
        while len(node_queue) > 0 and cursor < len(genotype):
            current_tree = node_queue.pop(0)
            _, op_ar = current_tree[0]

            for i in range(op_ar):
                cursor += 1
                if cursor < len(genotype):
                    if cursor < self.op_size:
                        op_code = int(genotype[cursor])
                        op_node = self.operators[op_code]
                        new_child = [op_node]
                        current_tree.append(new_child)
                        node_queue.append(new_child)
                    else:
                        op_node = genotype[cursor], 0
                        new_child = [op_node]
                        current_tree.append(new_child)

        return expr_tree

    def _generate_expression(self, expr_tree):
        """
        Transforms an expression tree into a string for the formula with infix notation.
        """

        final_str = ""
        if len(expr_tree) == 3:
            final_str = "(" + self._generate_expression(expr_tree[1]) + str(expr_tree[0][0]) + self._generate_expression(expr_tree[2]) + ")"
        elif len(expr_tree) == 1:
            final_str = str(expr_tree[0][0])
        else:
            final_str = str(expr_tree[0][0]) + "(" + ",".join([self._generate_expression(i) for i in expr_tree[1:]]) + ")"

        return final_str

    def decode(self, genotype):
        """
        Transforms a vector into an expression string.
        """

        expr_tree = self._construct_tree(genotype)
        formula = self._generate_expression(expr_tree)

        return formula


class GEPPSMEncoding(GEPEncoding):
    def __init__(self, max_size, op_size, input_dim, n_params, operators=None):
        if operators is None:
            operators = [("+", 2), ("-", 2), ("*", 2), ("/", 2)]
            operators += [(f"p_{i}", 0) for i in range(n_params)]
            operators += [(f"x_{i}", 0) for i in range(input_dim)]
        super().__init__(max_size, op_size, operators)
        self.input_dim = input_dim
        self.n_params = n_params

    def decode(self, genotype):
        genotype_mod = genotype.copy().astype(object)
        genotype_int = genotype.astype(int)
        for idx, val in enumerate(genotype_int[self.op_size :]):
            adj_idx = idx + self.op_size
            if val >= 0:
                genotype_mod[adj_idx] = f"x_{val%self.input_dim}"
            else:
                genotype_mod[adj_idx] = f"p_{(-val-1)%self.n_params}"

        return super().decode(genotype_mod)


class EvalGEPModel(ObjectiveVectorFunc):
    def __init__(self):
        pass

    def objective(self, eq_str):
        pass


def print_tree(tree, level=0):
    print(level * "  " + str(tree[0][0]))
    if len(tree) == 3:
        print_tree(tree[1], level + 1)
        print_tree(tree[2], level + 1)


if __name__ == "__main__":
    import sympy

    gpeenc = GEPPSMEncoding(15, 7, 4, 4)

    # encoded = np.array([1,2,0,2,4,2,3,-1,-1,-2,-3,1,-4,4,-1])
    encoded = np.random.randint(-4, 4, size=15)
    tree = gpeenc.decode(encoded)
    print(type(tree))

    equation = sympy.parsing.sympy_parser.parse_expr(tree)

    print(equation)
    for i in range(4):
        print(f"partial[p_{i}](f) = ", sympy.diff(equation, f"p_{i}"))
