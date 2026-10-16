"""Pinned VERL hooks: deterministic initialization and an importable dataset class."""
import os
import random

import numpy as np
import torch
from math_rl.completion_dataset import CompletionDataset

seed = int(os.environ.get("MATH_RL_SEED", "42"))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
