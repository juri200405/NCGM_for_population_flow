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
        Z = theta.mul(y.unsqueeze(1))

        return Z, theta
    
class NCGM_objective(nn.Module):
    def __init__(self, location_size, neighbor_size):
        super(NCGM_objective, self).__init__()

        self.L = location_size
        self.nei = neighbor_size
    
    def forward(self, Z, theta, yt, yt1, lam):
        L = Z.mm(theta.log().add(-1, Z.log()).add(1)).t().sum(1)

        e = yt.add(-1, Z.sum(1)).pow(2).add(1, yt1.add(-1, Z.sum(0).t()).pow(2))

        G = L.add(-1 * lam, e)

        return G.sum().neg()