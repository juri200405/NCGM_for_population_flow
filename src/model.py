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

        self.Z = torch.randn(self.T, self.L, self.L, requires_grad=True)
        self.one = torch.Tensor.new_ones(self.T, self.L, self.L, dtype=torch.float)
    
    def forward(self, input, y, lam):
        L = torch.sum(torch.mul(self.Z, self.one - torch.log(self.Z) + torch.log(self.f(input))))
        G = L - lam * (torch.sum((y - torch.sum(self.Z, 2) ** 2)) + torch.sum((y - torch.sum(self.Z, 1)) ** 2))
        return G * (-1)
    
    def f(self, inputs):
        out = torch.tanh(self.fc1(inputs))
        out = self.fc2(out)
        out = self.softmax(out)
        return out


if __name__ == "__main__":
    model = NCGM(5, 8)
    input_tensor = torch.randn(3, 6, 8, 5)
    output_tensor = model(input_tensor)
    print(output_tensor)