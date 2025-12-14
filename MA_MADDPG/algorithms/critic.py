import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super(Critic, self).__init__()
        self.f1 = nn.Linear(input_dim, hid_dim)
        self.f2 = nn.Linear(hid_dim, hid_dim)
        self.f3 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = torch.relu(self.f1(x))
        x = torch.relu(self.f2(x))
        x = self.f3(x)
        return x
