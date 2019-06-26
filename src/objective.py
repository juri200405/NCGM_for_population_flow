import torch
import torch.nn as nn

class Objevtive(nn.Module):
    def __init__(self, time_size, location_size, N_l):
        super(Objevtive, self).__init__()

        self.T = time_size
        self.L = location_size
        self.N_l = N_l
        self.lam = 0.5
    
    def forward(self, y, z, theta):
        log_likelihood = 0.0
        log_z = torch.log(z)
        log_theata = torch.log(theta)
        for t in range(self.T - 1):
            for l in range(self.L):
                for ll in range(len(N_l[l])):
                    log_likelihood += z[t][l][self.N_l[l][ll]] * (1 - log_z[t][l][self.N_l[l][ll]] + log_theata[t][l][ll])
        G = log_likelihood - self.lam * (torch.sum(y - torch.sum(z, 2)) + torch.sum(y - torch.sum(z, 1)))[0]
        return G


if __name__ == "__main__":
    