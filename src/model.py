import torch
import torch.nn as nn

class NCGM(nn.Module):
    def __init__(self, input_size, hidden_size, time_size, location_size, neighbor_size):
        super(NCGM, self).__init__()

        self.time_size = time_size
        self.input_size = input_size
        self.hid_size = hidden_size
        self.L = location_size
        self.nei = neighbor_size

        self.fc1 = nn.Linear(self.input_size, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 1)
        self.softmax = nn.Softmax(2)

        self.Z = nn.Parameter(torch.ones(self.time_size - 1, self.L, self.nei))

    def forward(self, input, y):
        out = self.fc1(input).tanh()
        out = self.fc2(out)
        out = self.softmax(out)

        theta = out.squeeze()
        '''
        for t in range(self.time_size - 1):
            for i in range(self.L):
                m = torch.distributions.multinomial.Multinomial(total_count=int(y[t,i]), probs=theta[t,i])
                self.Z[t,i,:] = m.sample()
        '''
        return theta
    
class NCGM_objective(nn.Module):
    def __init__(self, time_size, location_size, neighbor_size, device):
        super(NCGM_objective, self).__init__()

        self.time_size = time_size
        self.L = location_size
        self.nei = neighbor_size
        self.device = device

        self.mse_loss_t = nn.MSELoss(reduction='sum')
        self.mse_loss_t1 = nn.MSELoss(reduction='sum')
    
    def forward(self, theta, Z, y, lam):
        theta_log = theta.clamp(min=3.7835e-44).log()
        Z_log = Z.clamp(min=3.7835e-44).log()

        obj_L = Z.mul(theta_log.add(1).add(-1, Z_log)).sum()

        yt = y.narrow(0, 0, self.time_size - 1)
        yt1 = y.narrow(0, 1, self.time_size - 1)
        et = self.mse_loss_t(yt, Z.transpose(1,2).sum(1))
        et1 = self.mse_loss_t1(yt1, Z.sum(1))

        G = obj_L.add(-1 * lam, et.add(1, et1))

        return G.neg()