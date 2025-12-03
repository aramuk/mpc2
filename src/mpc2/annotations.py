from typing import Dict, List, Tuple, Union

import crypten
import torch

TensorType = Union[torch.Tensor, crypten.mpc.MPCTensor]
State = Dict
Action = TensorType
Reward = float
Trajectory = List[Action]