import torch
import torch.nn as nn
import torch.distributed

class NCGM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hid_size = hidden_size

        self.fc1 = nn.Linear(self.input_size, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, 1)
        self.softmax = nn.Softmax(0)
    
    def forward(self, input):
        out = torch.tanh(self.fc1(input))
        out = self.fc2(out)
        out = self.softmax(out)
        return out

if __name__ == "__main__":
    model = NCGM(5, 8)
    input_tensor = torch.randn(2, 5)
    output_tensor = model(input_tensor)
    print(output_tensor)

    tmp = torch.randn(3, 3, 3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                tmp[i][j][k] = (j + 1) * (10 ** i)
    print(torch.sum(tmp, 0))
    print(torch.sum(tmp, 2))
    print(torch.sum(tmp, 0) - torch.sum(tmp, 2))

    m = torch.distributions.multinomial.Multinomial(100, output_tensor.squeeze())
    x = m.sample()
    print(x)
    