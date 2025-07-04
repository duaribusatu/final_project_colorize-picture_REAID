import torch
import torch.nn as nn

class BaseColor(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_cent = 50.
        self.l_norm = 100.
        self.ab_norm = 110.

    def normalize_l(self, in_l: torch.Tensor) -> torch.Tensor:
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_l(self, in_l: torch.Tensor) -> torch.Tensor:
        return in_l * self.l_norm + self.l_cent

    def normalize_ab(self, in_ab: torch.Tensor) -> torch.Tensor:
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab: torch.Tensor) -> torch.Tensor:
        return in_ab * self.ab_norm
