
import numpy as np
import torch
from torch import nn

from torch.nn import functional as F


class Model(nn.Module):
    """
    ## Model
    """

    def __init__(self, state_size, action_size):
        super().__init__()

        self.fc1 = nn.Linear(in_features=state_size, out_features=256)
        nn.init.orthogonal_(self.fc1.weight, np.sqrt(0.01))
        self.fc2 = nn.Linear(in_features=256, out_features=256)
        nn.init.orthogonal_(self.fc2.weight, np.sqrt(0.01))
        self.fc3 = nn.Linear(in_features=256, out_features=256)
        nn.init.orthogonal_(self.fc3.weight, np.sqrt(0.01))

        # policy output
        self.pi_logits = nn.Linear(in_features=256, out_features=action_size)
        nn.init.orthogonal_(self.pi_logits.weight, np.sqrt(0.01))

        # value function output
        self.value = nn.Linear(in_features=256, out_features=1)
        nn.init.orthogonal_(self.value.weight, 1)

    def forward(self, obs):
        h: torch.Tensor

        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))

        value = self.value(h).reshape(-1)

        return self.pi_logits(h), value
