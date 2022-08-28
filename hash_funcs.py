import torch
import abc


class Hash(object):
    def __init__(self, nclusters):
        self.nclusters = nclusters

    @abc.abstractmethod
    def learn(self, observations: torch.Tensor):
        return

    @abc.abstractmethod
    def encode(self, observations: torch.Tensor):
        codes = torch.zeros((len(observations), 1), device=observations.device)
        return codes
