import csv
from pathlib import Path

import torch
import torch.optim as optim

import model

if __name__ == "__main__":
    with open(str(Path("datas") / "sample_population.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        population_data = [[int(col) for col in row] for row in reader]
    
    with open(str(Path("datas") / "sample_neighbor.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        neighbor = [[int(col) for col in row] for row in reader]
    
    location_table = [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]
    adj_table = [[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    torch.set_default_dtype(torch.double)


    mod = model.NCGM(5, 8, 8, 4)
    mod.to(device)

    optimizer = optim.SGD(mod.parameters(), lr=0.01)

    input_list = [[[[t / 7.0 - 0.5, location_table[l][0], location_table[l][1], location_table[ll][0] - location_table[l][0], location_table[ll][1] - location_table[l][1]] for ll in range(4)] for l in range(4)] for t in range(8)]
    input_tensor = torch.Tensor(input_list)
    input_tensor.to(device)

    population_tensor = torch.Tensor(population_data)
    population_tensor.to(device)

    adj_tensor = torch.Tensor(adj_table)
    adj_tensor.to(device)

    mod.train()
    for i in range(1000):
        output_tensor = mod(input_tensor, population_tensor, adj_tensor, 1.0)
        if i % 100 == 0:
            print(output_tensor)

        optimizer.zero_grad()
        output_tensor.backward()
        optimizer.step()
    print(torch.exp(mod.Z_dash))