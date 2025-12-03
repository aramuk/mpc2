import crypten
import torch

from .types import Tensor

def tensor_norm(tensor: Tensor) -> Tensor:
    if isinstance(tensor, torch.Tensor):
        return torch.norm(tensor)
    elif isinstance(tensor, crypten.mpc.MPCTensor):
        return crypten.mpc.norm(tensor)
    else:
        raise ValueError(f"Unsupported tensor type: {type(tensor)}")