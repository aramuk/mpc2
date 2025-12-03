import os
from typing import Any, List, Type

import crypten
import numpy as np
import torch

from mpc2.annotations import TensorType

def set_tensor_type(ttype: Type[TensorType]):
    if isinstance(ttype, torch.Tensor):
        os.environ["MPC2_TENSOR_TYPE"] = "torch"
    elif isinstance(ttype, crypten.mpc.MPCTensor):
        os.environ["MPC2_TENSOR_TYPE"] = "crypten"
    elif isinstance(ttype, np.ndarray):
        os.environ["MPC2_TENSOR_TYPE"] = "numpy"
    else:
        raise ValueError(f"Unsupported tensor type: {type(ttype)}")

def tensor_norm(tensor: TensorType) -> TensorType:
    if isinstance(tensor, torch.Tensor):
        return torch.norm(tensor)
    elif isinstance(tensor, crypten.mpc.MPCTensor):
        return crypten.mpc.norm(tensor)
    elif isinstance(tensor, np.ndarray):
        return np.linalg.norm(tensor)
    else:
        raise ValueError(f"Unsupported tensor type: {type(tensor)}")

def make_tensor(*args, **kwargs) -> TensorType:
    if os.environ["MPC2_TENSOR_TYPE"] == "torch":
        return torch.tensor(*args, **kwargs)
    elif os.environ["MPC2_TENSOR_TYPE"] == "crypten":
        return crypten.mpc.MPCTensor(*args, **kwargs)
    elif os.environ["MPC2_TENSOR_TYPE"] == "numpy":
        return np.array(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported tensor type: {os.environ['MPC2_TENSOR_TYPE']}")

def get_tensor_type() -> Type[TensorType]:
    if os.environ["MPC2_TENSOR_TYPE"] == "torch":
        return torch.Tensor
    elif os.environ["MPC2_TENSOR_TYPE"] == "crypten":
        return crypten.mpc.MPCTensor
    elif os.environ["MPC2_TENSOR_TYPE"] == "numpy":
        return np.ndarray
    else:
        raise ValueError(f"Unsupported tensor type: {os.environ['MPC2_TENSOR_TYPE']}")

def get_tensor_library() -> Any:
    if os.environ["MPC2_TENSOR_TYPE"] == "torch":
        return torch
    elif os.environ["MPC2_TENSOR_TYPE"] == "crypten":
        return crypten.mpc
    elif os.environ["MPC2_TENSOR_TYPE"] == "numpy":
        return np
    else:
        raise ValueError(f"Unsupported tensor type: {os.environ['MPC2_TENSOR_TYPE']}")