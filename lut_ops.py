import torch
from torch import nn

import quantizers
import lookup_tables


class LinearPQSTE(nn.Module):
    def __init__(self, linear_layer: nn.Linear, pq: quantizers.ProductQuantizer):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.weight = nn.Parameter(linear_layer.weight.detach())
        self.bias = False
        if linear_layer.bias:
            self.bias = nn.Parameter(linear_layer.bias.detach())
        self.pq = pq
        self.amm_training = False
        self.optimizing = False

    def forward(self, x):
        if self.amm_training:
            assert not self.training
            self.pq.quantize(x)
            out = x @ self.weight.t()
            if self.bias:
                out = out + self.bias
        if self.optimizing:
            assert self.training
            out = lookup_tables.LinearPQTable.apply(x, self.weight, self.bias)

