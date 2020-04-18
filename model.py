import torch
import torch.nn as nn
import torch.nn.functional as f


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, duel=0, f1_unit=64, f2_unit=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.duel = duel

        self.f1 = nn.Linear(state_size, f1_unit)
        self.f2 = nn.Linear(f1_unit, f2_unit)
        self.f3 = nn.Linear(f2_unit, action_size)

        self.state_value = nn.Linear(f2_unit, 1)

    def forward(self, state):
        st = f.relu(self.f1(state))
        st = f.relu(self.f2(st))
        if self.duel:
            return self.f3(st) + self.state_value(st)
        else:
            return self.f3(st)
