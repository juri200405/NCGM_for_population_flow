import torch
import torch.nn as nn

class NCGM(nn.Module):
    def __init__(self, input_size, hidden_size, time_size, location_size):
        super().__init__()

        self.input_size = input_size
        self.hid_size = hidden_size
        self.T = time_size
        self.L = location_size

        self.fc1 = nn.Linear(self.input_size, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 1)
        self.softmax = nn.Softmax(2)

        #self.Z = nn.Parameter(torch.randint(1, 14, (self.T - 1, self.L, self.L), requires_grad=True, dtype=torch.double))
        self.Z_dash = nn.Parameter(torch.log(torch.randint(1, 14, (self.T - 1, self.L, self.L), dtype=torch.double)))
        self.one = torch.ones(self.T - 1, self.L, self.L, dtype=torch.double)
    
    def forward(self, input, y, adj, lam):
        Z = torch.exp(self.Z_dash)
        print(Z.size())
        adj_Z = torch.mul(self.Z_dash, adj)
        print(adj_Z.size())
        lf = torch.log(self.f(torch.narrow(input, 0, 0, self.T - 1)).squeeze())
        print(lf.size())
        tmp = self.one - adj_Z + lf

        L = torch.sum(torch.mul(torch.exp(adj_Z), tmp))
        y_b = torch.narrow(y, 0, 0, self.T - 1)
        y_a = torch.narrow(y, 0, 1, self.T - 1)
        e_b = torch.diagonal(Z * adj, 0, 1, 2)
        e_a = torch.diagonal(torch.transpose(Z, 1, 2) * adj, 0, 1, 2)
        G = L - lam * (torch.sum((y_b - e_b) ** 2) + torch.sum((y_a - e_a) ** 2))
        return G * (-1)
    
    def f(self, inputs):
        out = torch.tanh(self.fc1(inputs))
        out = self.fc2(out)
        out = self.softmax(out)
        return out