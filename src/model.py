import torch
import torch.nn as nn

class NCGM(nn.Module):
    def __init__(self, input_size, hidden_size, time_size, location_size, neighbor_size):
        super(NCGM, self).__init__()

        self.input_size = input_size
        self.hid_size = hidden_size
        self.L = location_size
        self.nei = neighbor_size
        self.time_size = time_size

        self.fc1 = nn.Linear(self.input_size, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 1)
        self.softmax = nn.Softmax(1)

        self.Z = nn.Parameter(torch.ones(self.time_size - 1, self.L, self.nei))

    def forward(self, input):
        out = self.fc1(input).tanh()
        out = self.fc2(out)
        out = self.softmax(out)

        theta = out.squeeze()

        return theta
    
class NCGM_objective(nn.Module):
    def __init__(self, location_size, neighbor_size):
        super(NCGM_objective, self).__init__()

        self.L = location_size
        self.nei = neighbor_size

        self.mse_loss_t = nn.MSELoss(reduction='sum')
        self.mse_loss_t1 = nn.MSELoss(reduction='sum')
    
    def forward(self, theta, Z, yt, yt1, lam):
        theta_log = theta.clamp(min=3.7835e-44).log()
        Z_log = Z.clamp(min=3.7835e-44).log()

        obj_L = Z.mul(theta_log.add(1).add(-1, Z_log)).sum()

        et = self.mse_loss_t(yt, Z.t().sum(0))
        et1 = self.mse_loss_t1(yt1, Z.sum(0))

        G = obj_L.add(-1 * lam, et.add(1, et1))

        return G.neg()