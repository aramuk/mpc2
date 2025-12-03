from typing import Dict, List, Tuple, Union

import crypten
import torch

Tensor = Union[torch.Tensor, crypten.mpc.MPCTensor]
State = Dict
Action = Tensor
Reward = float
Trajectory = List[Tuple[State, Action, Reward]]