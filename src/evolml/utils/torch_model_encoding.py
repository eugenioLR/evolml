from __future__ import annotations
import numpy as np
import torch
from torch import nn
import metaheuristic_designer as mhd
from copy import copy, deepcopy


class PytorchModelEncoding(mhd.Encoding):
    def __init__(self, nn_model):
        super().__init__()
        self.nn_model = nn_model

    def encode(self, phenotype: nn.Module):
        return torch.nn.utils.parameters_to_vector(phenotype.parameters())

    def decode(self, genotype: np.ndarray):
        new_model = deepcopy(self.nn_model)
        torch.nn.utils.vector_to_parameters(torch.Tensor(genotype), new_model.parameters())
        return new_model


class PytorchModelInitializer(mhd.Initializer):
    def __init__(self, nn_model, pop_size=1, encoding=None):
        super().__init__(pop_size, encoding)
        if isinstance(nn_model, nn.Module):
            nn_model = type(nn_model)
        self.nn_model = nn_model

    @torch.no_grad()
    def generate_random(self, objfunc):
        param_vec = np.asarray(torch.nn.utils.parameters_to_vector(self.nn_model().parameters()))
        return mhd.Individual(objfunc, param_vec, encoding=self.encoding)
    
    def generate_individual(self, objfunc):
        return self.generate_random(objfunc)


if __name__ == "__main__":
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.f1 = nn.Linear(10, 10)
            self.f2 = nn.Linear(10, 20)
            self.f3 = nn.Linear(20, 15)
            self.f4 = nn.Linear(15, 10)
            self.f5 = nn.Linear(10, 1)
        
        def forward(self, x):
            x = self.ReLU(self.f1(x))
            x = self.ReLU(self.f2(x))
            x = self.ReLU(self.f3(x))
            x = self.ReLU(self.f4(x))
            return self.f5(x)
    
    model = MLP()

    torch_encoding = PytorchModelEncoding(model)
    params = torch_encoding.encode(model)
    print(params, params.shape)

    new_model = torch_encoding.decode(params)
    print(new_model)
