from __future__ import annotations
from typing import Iterable
import numpy as np
import torch
from torch import nn
import metaheuristic_designer as mhd
from copy import deepcopy


class PytorchModelEncoding(mhd.Encoding):
    def __init__(self, nn_model):
        super().__init__(vectorized=False)

        self.nn_model = nn_model

    def encode_func(self, solution: nn.Module) -> np.ndarray:
        return torch.nn.utils.parameters_to_vector(solution.parameters())

    def decode_func(self, indiv: np.ndarray) -> nn.Module:
        new_model = deepcopy(self.nn_model)
        torch.nn.utils.vector_to_parameters(torch.Tensor(indiv), new_model.parameters())
        return new_model


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
