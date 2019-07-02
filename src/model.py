import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

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

        self.Z = torch.randint(1, 14, (self.T - 1, self.L, self.L), requires_grad=True, dtype=torch.double)
        self.one = torch.ones(self.T - 1, self.L, self.L, dtype=torch.double)
    
    def forward(self, input, y, lam):
        lz = torch.log2(self.Z)
        lf = torch.log2(self.f(torch.narrow(input, 0, 0, self.T - 1)).squeeze())
        tmp = self.one - lz + lf

        L = torch.sum(torch.mul(self.Z, tmp))
        y_b = torch.narrow(y, 0, 0, self.T - 1)
        y_a = torch.narrow(y, 0, 1, self.T - 1)
        G = L - lam * (torch.sum((y_b - torch.sum(self.Z, 2)) ** 2) + torch.sum((y_a - torch.sum(self.Z, 1)) ** 2))
        return G * (-1)
    
    def f(self, inputs):
        out = torch.tanh(self.fc1(inputs))
        out = self.fc2(out)
        out = self.softmax(out)
        return out


if __name__ == "__main__":
    with open(str(Path("datas") / "sample_population.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        population_data = [[int(col) for col in row] for row in reader]
    
    with open(str(Path("datas") / "sample_neighbor.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        neighbor = [[int(col) for col in row] for row in reader]
    
    location_table = [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    torch.set_default_dtype(torch.double)


    model = NCGM(5, 8, 8, 4)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.05)

    input_list = [[[[t / 7.0 - 0.5, location_table[l][0], location_table[l][1], location_table[ll][0] - location_table[l][0], location_table[ll][1] - location_table[l][1]] for ll in range(4)] for l in range(4)] for t in range(8)]
    input_tensor = torch.Tensor(input_list)
    input_tensor.to(device)

    population_tensor = torch.Tensor(population_data)
    population_tensor.to(device)

    model.train()
    for i in range(1):
        output_tensor = model(input_tensor, population_tensor, 10.0)
        if i % 100 == 0:
            print(output_tensor)

        optimizer.zero_grad()
        output_tensor.backward()
        optimizer.step()