import torch
import quantizers
from torch import nn


class LinearPQTable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor,
                weight: torch.Tensor, bias: torch.Tensor,
                pq: quantizers.ProductQuantizer):
        ctx.save_for_backward(input, weight, bias)
        codes = pq.assign(input)
        tables = []
        for isub, vq in enumerate(pq.vqs):
            assert len(vq.prototypes.size()) == 2
            sub_weight = pq.space_divider.get_subvecs(weight, isub)
            table = vq.prototypes @ sub_weight.t()
            if bias:
                sub_bias = pq.space_divider.get_subvecs(bias.view(1, -1), isub)
                table += sub_bias.unsqueeze(0).expand_as(table)
            tables.append(table)
        output = torch.zeros((len(input), weight.size(0)), device=input.device)
        for isub, vq in enumerate(pq.vqs):
            lut_eles = table[isub][codes[:, isub]]
            output += lut_eles
        return output




