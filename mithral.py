import torch

import hash_utils
import quantizers
import lut_ops

from torch import nn
from hash_funcs import Hash
from hash_utils import Bucket


class MaddnessBucket(Bucket):
    def __init__(self):
        super().__init__()


class MaddnessHash(Hash):
    def __init__(self, nsplits=4):
        self.nsplits = nsplits
        nclusters = 2 ** nsplits
        super().__init__(nclusters)
        self.split_dims = None
        self.split_vals = None

    def learn(self, observations: torch.Tensor):
        N = len(observations)
        X = observations.view(N, -1)
        D = X.size(1)
        self.split_dims = [3 for _ in range(self.nclusters)]
        self.split_vals = [[2.5, 3.5] for _ in range(self.nclusters)]
        buckets = [MaddnessBucket() for _ in range(self.nclusters)]
        return

    def encode(self, observations: torch.Tensor):
        codes = torch.zeros((len(observations), 1), device=observations.device)
        return codes


class MaddnessVQ(quantizers.CentroidVQ):
    def __init__(self, nsplits):
        maddness_hash = MaddnessHash(nsplits=nsplits)
        super().__init__(maddness_hash)


class MaddnessPQ(quantizers.ProductQuantizer):
    def __init__(self, ncodebooks, nsplits):
        space_divider = hash_utils.Equal1dDivider(target_nsubspaces=2**nsplits)
        vqs = [MaddnessVQ(nsplits) for _ in range(ncodebooks)]
        super().__init__(vqs=vqs, space_divider=space_divider)


class MaddnessLinear(lut_ops.LinearPQSTE):
    def __init__(self, ncodebooks, nsplits, linear_layer: nn.Linear):
        pq = MaddnessPQ(ncodebooks, nsplits)
        super().__init__(pq=pq, linear_layer=linear_layer)