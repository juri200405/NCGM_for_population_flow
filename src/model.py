import torch
import torch.nn as nn

class NCGM(nn.Module):
    def __init__(self, input_size, hidden_size, time_size, location_size, neighbor_size):
        super(NCGM, self).__init__()

        #入力ベクトルの次元数
        self.input_size = input_size
        #隠れ層の次元数
        self.hid_size = hidden_size
        #セルの数
        self.L = location_size
        #セルに隣接するセルの数
        self.nei = neighbor_size
        #タイムステップ数
        self.time_size = time_size

        #θを求めるネットワーク
        self.fc1 = nn.Linear(self.input_size, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 1)
        self.softmax = nn.Softmax(1)

        #移動人数
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

        #セルの数
        self.L = location_size
        #セルに隣接するセルの数
        self.nei = neighbor_size

        self.mse_loss_t = nn.MSELoss(reduction='sum')
        self.mse_loss_t1 = nn.MSELoss(reduction='sum')
    
    def forward(self, theta, Z, yt, yt1, lam):
        #最小値以下の要素を最小値に置き換え
        theta_log = theta.clamp(min=3.7835e-44).log()
        Z_log = Z.clamp(min=3.7835e-44).log()

        #L'を求める
        obj_L = Z.mul(theta_log.add(1).add(-1, Z_log)).sum()

        #セルからの移動人数の制限
        et = self.mse_loss_t(yt, Z.t().sum(0))
        #セルへの移動人数の制限
        et1 = self.mse_loss_t1(yt1, Z.sum(0))

        #Gを求める
        G = obj_L.add(-1 * lam, et.add(1, et1))

        #Gを最大化するために、目的関数はGに-1をかけたものとする
        return G.neg()