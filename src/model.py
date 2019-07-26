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

    def forward(self, input):
        out = self.fc1(input).tanh()
        out = self.fc2(out)
        out = self.softmax(out)

        theta = out.squeeze()

        return theta
    
class NCGM_objective(nn.Module):
    def __init__(self, location_size, neighbor_size, device):
        super(NCGM_objective, self).__init__()

        self.L = location_size
        self.nei = neighbor_size
        self.device = device

        self.mse_loss = nn.MSELoss(reduction='sum')
    
    def forward(self, theta, yt, yt1, lam):
        theta_list = list(theta.unbind(0))
        theta_log_list = list(theta.log().clamp(min=-104.0).unbind(0))
        Z_list = []
        Z_log_list = []
        obj_L = torch.tensor(0.0, device=self.device)
        for i in range(self.L):
            m = torch.distributions.multinomial.Multinomial(total_count=int(yt[i]), probs=theta_list[i])
            Z = m.sample()
            Z_list.append(Z.unsqueeze(dim=0))
            Z_log_list.append(Z.log().clamp(min=-104.0))
            Ls_right = theta_log_list[i].add(1).add(-1, Z_log_list[i])
            obj_L = obj_L.add(Z.dot(Ls_right))
        
        Z_tensor = torch.cat(Z_list, 0)
        et = self.mse_loss(yt, Z_tensor.t().sum(0))
        et1 = self.mse_loss(yt1, Z_tensor.sum(0))

        G = obj_L.add(-1 * lam, et.add(1, et1))

        return G.neg(), Z_tensor