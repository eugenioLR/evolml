from __future__ import annotations
from typing import Iterable
import numpy as np
import torch
from torch import nn
import metaheuristic_designer as mhd
from copy import copy, deepcopy


class PytorchModelEncoding(mhd.Encoding):
    def __init__(self, nn_model):
        super().__init__()
        self.nn_model = nn_model

    def encode(self, population: Iterable[nn.Module]) -> np.ndarray:
        param_list = []
        for indiv in population:
            params = torch.nn.utils.parameters_to_vector(indiv.parameters())
            param_list.append(params)
        
        return np.array(params)

    def decode(self, genotype: np.ndarray) -> Iterable[nn.Module]:
        model_list = []
        for i in genotype:
            new_model = deepcopy(self.nn_model)
            torch.nn.utils.vector_to_parameters(torch.Tensor(i), new_model.parameters())
            model_list.append(new_model)
        
        return model_list


class PytorchModelInitializer(mhd.Initializer):
    def __init__(self, nn_model, pop_size=1, encoding=None, **kwargs):
        super().__init__(pop_size, encoding)
        if isinstance(nn_model, nn.Module):
            nn_model = type(nn_model)
        self.nn_model = nn_model

        self.model_kwargs = kwargs

    @torch.no_grad()
    def generate_random(self):
        new_model = self.nn_model(**self.model_kwargs).parameters()
        param_vec = torch.nn.utils.parameters_to_vector(new_model).numpy()

        return param_vec
