import torch
import torch.nn as nn



class Actor(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.rnn = torch.nn.GRU(
            input_size=hid_dim,
            hidden_size=hid_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)
        self.hidden = None

    """This is confusing that the hidden unit (self.hidden) is not used as the input, with which
    the rnn layer actually serves as normal non-linear layer. Also, with rnn, experience replay does not work, and gradient 
    should be updated sequentially. According to the provided code, we can replay rnn with a simple nn.linear.......
    Note that the obs is obs_his * obs_each_time. If we want to make the rnn work, let the input be (batch=1, obs_his_len, obs_dim) instead of (
    batch=1, 1, obs_his_len*obs_dim).
    """
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x, self.hidden = self.rnn(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
