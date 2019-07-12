import torch
import torch.nn as nn

class NCGM(nn.Module):
    def __init__(self, input_size, hidden_size, location_size, neighbor_size):
        super(NCGM, self).__init__()

        self.input_size = input_size
        self.hid_size = hidden_size
        self.L = location_size
        self.nei = neighbor_size

        self.fc1 = nn.Linear(self.input_size, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 1)
        self.softmax = nn.Softmax(1)

    def forward(self, input, y):
        out = self.fc1(input).tanh()
        out = self.fc2(out)
        out = self.softmax(out)

        theta = out.squeeze()
        Z = theta.mul(y.unsqueeze(1)).log().clamp(min=-104.0)

        return Z, theta
    
class NCGM_objective(nn.Module):
    def __init__(self, location_size, neighbor_size):
        super(NCGM_objective, self).__init__()

        self.L = location_size
        self.nei = neighbor_size
    
    def forward(self, Z, theta, yt, yt1, lam):
        log_theta = theta.log().clamp(min=-104.0)
        log_theta_add_1 = log_theta.add(1)
        Ls_right = log_theta_add_1.add(-1, Z)

        L = Z.exp().mul(Ls_right).sum(1)

        et = yt.add(-1, Z.exp().sum(1)).pow(2)
        et1 = yt1.add(-1, Z.exp().sum(1)).pow(2)

        G = L.add(-1 * lam, et.add(1, et1))

        return G.sum().neg()