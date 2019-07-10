import torch
import torch.nn as nn

class NCGM(nn.Module):
    def __init__(self, input_size, hidden_size, time_size, location_size, neighbor_size):
        super().__init__()

        self.input_size = input_size
        self.hid_size = hidden_size
        self.T = time_size
        self.L = location_size
        self.nei = neighbor_size

        self.fc1 = nn.Linear(self.input_size, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 1)
        self.softmax = nn.Softmax(2)

        #self.Z = nn.Parameter(torch.randint(1, 14, (self.T - 1, self.L, self.L), requires_grad=True, dtype=torch.double))
        #self.Z_dash = nn.Parameter(torch.log(torch.randint(1, 14, (self.T - 1, self.L, self.L), dtype=torch.double)))
        self.one = torch.ones(self.T - 1, self.L, self.L, dtype=torch.double, requires_grad=True)
    
    def forward(self, input, y):
        '''
        Z = torch.exp(self.Z_dash)
        adj_Z = torch.mul(self.Z_dash, adj)
        lf = torch.log(self.f(torch.narrow(input, 0, 0, self.T - 1)).squeeze())
        tmp = self.one - adj_Z + lf

        L = torch.sum(torch.mul(torch.exp(adj_Z), tmp))
        y_b = torch.narrow(y, 0, 0, self.T - 1)
        y_a = torch.narrow(y, 0, 1, self.T - 1)
        e_b = torch.diagonal(Z * adj, 0, 1, 2)
        e_a = torch.diagonal(torch.transpose(Z, 1, 2) * adj, 0, 1, 2)
        G = L - lam * (torch.sum((y_b - e_b) ** 2) + torch.sum((y_a - e_a) ** 2))
        return G * (-1)
        '''
        theta = self.f(torch.narrow(input, 0, 0, self.T - 1)).squeeze()
        #Z = torch.zeros(self.T - 1, self.L, self.nei, dtype=torch.double)
        #Z = torch.zeros(self.T - 1, self.L, self.L, dtype=torch.double)
        y_list = []
        for _ in range(self.L):
            y_list.append(torch.unsqueeze(torch.narrow(y, 0, 0, self.T - 1), 2))
        ys = torch.cat(y_list, 2)
        Z = torch.mul(theta, ys)
        '''
        for i in range(self.T - 1):
            for j in range(self.L):
                #x = torch.distributions.multinomial.Multinomial(y[i, j].item(), probs=theta[i, j]).sample()
                x = torch.mul(theta[i, j], y[i, j])
                #for k in range(self.nei):
                for k in range(self.L):
                    Z[i, j, k] = x[k]
        '''
        return Z, theta
    
    def f(self, inputs):
        out = torch.tanh(self.fc1(inputs))
        out = self.fc2(out)
        out = self.softmax(out)
        return out
    
    def objective_func(self, Z, theta, y, lam):
        '''
        L = 0.0
        for t in range(self.T - 1):
            for l in range(self.L):
                L += torch.dot(Z[t, l], (torch.ones(self.nei, dtype=torch.double) - torch.log(Z[t, l]) + theta[t, l])).item()
        
        for t in range(self.T - 1):
            for l in range(self.L):
                ztlld = 0.0
                ztldl = 0.0
                for ld in range(self.nei):
                    ztlld += Z[t, l, ld]
        '''
        L = torch.sum(torch.mul(Z, (self.one - torch.log(Z) + torch.log(theta))))

        y_b = torch.narrow(y, 0, 0, self.T - 1)
        y_a = torch.narrow(y, 0, 1, self.T - 1)
        e_b = torch.sum(Z, 2)
        e_a = torch.sum(torch.transpose(Z, 1, 2), 2)
        G = L - lam * (torch.sum((y_b - e_b) ** 2) + torch.sum((y_a - e_a) ** 2))

        return torch.neg(G)