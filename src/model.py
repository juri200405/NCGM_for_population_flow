import torch
import torch.nn as nn

class NCGM(nn.Module):
    def __init__(self, input_size, hidden_size, time_size, location_size, neighbor_size):
        super(NCGM, self).__init__()

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
        #self.one = torch.ones(self.T - 1, self.L, self.L, dtype=torch.double, requires_grad=True)
    
    def forward(self, input, y):
        theta = self.f(torch.narrow(input, 0, 0, self.T - 1)).squeeze()
        #Z = torch.zeros(self.T - 1, self.L, self.nei, dtype=torch.double)
        #Z = torch.zeros(self.T - 1, self.L, self.L, dtype=torch.double)
        
        '''
        for i in range(self.T - 1):
            for j in range(self.L):
                #x = torch.distributions.multinomial.Multinomial(y[i, j].item(), probs=theta[i, j]).sample()
                x = torch.mul(theta[i, j], y[i, j])
                #for k in range(self.nei):
                for k in range(self.L):
                    Z[i, j, k] = x[k]
        '''

        yt = torch.unsqueeze(torch.narrow(y, 0, 0, self.T - 1), 2)
        Z = torch.mul(theta, yt)

        return Z, theta
    
    def f(self, inputs):
        out = torch.tanh(self.fc1(inputs))
        out = self.fc2(out)
        out = self.softmax(out)
        return out

class NCGM_objective(nn.Module):
    def __init__(self, time_size, location_size):
        super(NCGM_objective, self).__init__()

        self.T = time_size
        self.L = location_size

        self.one = torch.ones(self.T - 1, self.L, self.L, dtype=torch.double, requires_grad=True)
    
    def forward(self, Z, theta, y, lam):
        #L = torch.sum(torch.mul(Z, (self.one - torch.log(Z) + torch.log(theta))))
        L = torch.sum(torch.mul(Z, torch.add(torch.add(torch.log(theta), 1), -1, torch.log(Z))))
        
        y_b = torch.narrow(y, 0, 0, self.T - 1)
        y_a = torch.narrow(y, 0, 1, self.T - 1)
        e_b = torch.sum(Z, 2)
        e_a = torch.sum(torch.transpose(Z, 1, 2), 2)

        d_b = torch.sum(torch.pow(torch.add(y_b, -1, e_b), 2))
        d_a = torch.sum(torch.pow(torch.add(y_a, -1, e_a), 2))

        G = torch.add(L, -1 * lam, torch.add(d_b, 1, d_a))

        return torch.neg(G)