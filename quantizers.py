import torch
import abc

from typing import List
from hash_utils import SpaceDivider
from hash_funcs import Hash


class VectorQuantizer(object):
    def __init__(self, hash_func: Hash):
        self.hash_func = hash_func
        self.prototypes = None

    @abc.abstractmethod
    def quantize(self, observations: torch.Tensor):
        self.hash_func.learn(observations)
        return

    def assign(self, observations: torch.Tensor):
        codes = self.hash_func.encode(observations=observations)
        return codes

    @property
    def nprototypes(self):
        return self.hash_func.nclusters


class CentroidVQ(VectorQuantizer):
    def __init__(self, hash_func: Hash):
        super().__init__(hash_func)

    def quantize(self, observations: torch.Tensor):
        buckets = self.hash_func.learn(observations)
        self.prototypes = torch.zeros(
            [self.nprototypes] + list(observations.size()[1:]))
        for ibuck, bucket in enumerate(buckets):
            self.prototypes[ibuck] = bucket.get_centroid(observations)
        return


class ProductQuantizer(object):
    def __init__(self, vqs: List[VectorQuantizer], space_divider: SpaceDivider):
        self.vqs = vqs
        self.space_divider = space_divider
        self.prototypes = None

    @property
    def nvqs(self):
        return len(self.vqs)

    def quantize(self, observations: torch.Tensor):
        self.space_divider.divide(observations=observations)
        for ivq, vq in enumerate(self.vqs):
            vq.quantize(
                observations=observations[self.space_divider.get_subvecs(
                    observations, isub=ivq)])
        return

    def assign(self, observations: torch.Tensor):
        observations_size = observations.size()
        N = observations_size[0]
        observation_size = observations_size[1:]
        assignments = torch.zeros((N, self.nvqs), device=observations.device)
        for ivq, vq in enumerate(self.vqs):
            subspace = self.space_divider.get_subvecs(observations, ivq)
            assignments[:, ivq] = vq.assign(observations=observations[subspace])
        return assignments

