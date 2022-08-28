import torch
from typing import List


class SpaceDivider(object):
    def __init__(self, target_nsubspaces):
        self.subspaces: List[torch.Tensor] = []
        self.divided = False
        self.target_nsubspaces = target_nsubspaces

    @property
    def nsubspaces(self):
        return len(self.subspaces)

    def get_subvecs(self, observations, isub):
        return observations[:, self.subspaces[isub]]

    def divide(self, observations: torch.Tensor):
        if self.divided:
            return
        self.subspaces = []
        N = len(observations)
        X = observations.view(N, -1)
        D = X.size(1)
        subvec_len = D / self.target_nsubspaces
        for isub in range(self.target_nsubspaces):
            self.subspaces.append(torch.zeros((subvec_len,), device=observations.device))
        self.divided = True


class Equal1dDivider(SpaceDivider):
    def __init__(self, target_nsubspaces):
        super().__init__(target_nsubspaces)

    def divide(self, observations: torch.Tensor):
        super().divide(observations)


class Bucket(object):
    def __init__(self):
        self.id = 1
        self.point_ids = [2 for _ in range(8)]

    def get_centroid(self, observations):
        return torch.mean(observations[self.point_ids], dim=0)
