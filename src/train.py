import csv
from pathlib import Path
import tqdm
from collections import OrderedDict

import torch
import torch.optim as optim

import model

def read_data(dpath, filename):
    with open(str(dpath / filename), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        population_data = []
        pop_tmp = []
        time = 0
        for row in reader:
            if time != int(row["time"]):
                population_data.append(pop_tmp)
                pop_tmp = []
                time = int(row["time"])
            pop_tmp.append(int(row["population"]))
        if pop_tmp != []:
            population_data.append(pop_tmp)
    
    with open(str(dpath / "chohu_adj.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        adj = [[int(col) for col in row] for row in reader]
    
    with open(str(dpath / "chohu_xy.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        xy = [[float(col) for col in row] for row in reader]
    
    with open(str(dpath / "chohu_adj_list.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        neighbor = [[int(col) for col in row] for row in reader]
    
    return population_data, adj, xy, neighbor


def read_samlpe():
    with open(str(Path("datas") / "sample_population.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        population_data = [[int(col) for col in row] for row in reader]
    
    with open(str(Path("datas") / "sample_neighbor.csv"), 'rt', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        neighbor = [[int(col) for col in row] for row in reader]
    
    location_table = [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]
    adj_table = [[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]

    return population_data, location_table, adj_table, neighbor

if __name__ == "__main__":
    '''
    p = torch.tensor([1.0, 1.0, 1.0, 1.0])
    x = torch.tensor([[1.0, 5.0, 1.0, 1.0], [5.0, 1.0, 1.0, 1.0]])
    m = torch.distributions.multinomial.Multinomial(total_count=10, probs=x)
    y = m.log_prob(x)
    print(y)
    print(torch.exp(y))
    print(m.sample())

    
    adj_list = []
    for row in range(117):
        tmp_list = []
        for col in range(117):
            if adj_table[row][col] == 1:
                tmp_list.append(str(col))
        adj_list.append(tmp_list)
    
    with open(str(Path("datas/chohu") / "chohu_adj_list.csv"), 'wt', encoding='utf-8') as csv_file:
        for i in range(117):
            csv_file.write(','.join(adj_list[i]))
            csv_file.write('\n')
    '''
    #neighbor_size = 10
    neighbor_size = 4
    #time_size = 9480
    time_size = 8
    #location_size = 117
    location_size = 4
    #population_data, adj_table, location_table, neighbor_table = read_data(Path("datas/chohu"), "chohu_01.csv")
    population_data, adj_table, location, neighbor_table = read_samlpe()
    #location = [[row[0] / 11 - 0.5, row[1] / 14 - 0.5] for row in location_table]
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    torch.set_default_dtype(torch.double)
    torch.set_grad_enabled(True)
    torch.autograd.set_detect_anomaly(True)


    mod = model.NCGM(5, 8, location_size, neighbor_size)
    mod.to(device)

    objective = model.NCGM_objective(location_size, neighbor_size)

    optimizer = optim.SGD(mod.parameters(), lr=0.01)

    input_list = []
    population_list = []
    for t in tqdm.trange(time_size):
        input_list_tmp = []
        for l in range(location_size):
            input_tmp = []
            #for ll in neighbor_table[l]:
            for ll in range(location_size):
                input_tmp.append([t / float(time_size) - 0.5, location[l][0], location[l][1], location[ll][0] - location[l][0], location[ll][1] - location[l][1]])
            while len(input_tmp) < neighbor_size:
                input_tmp.append([t / float(time_size) - 0.5, location[l][0], location[l][1], 0.0, 0.0])
            input_list_tmp.append(input_tmp)
        input_list.append(torch.tensor(input_list_tmp, device=device, dtype=torch.double))
        population_list.append(torch.tensor(population_data[t], dtype=torch.double, device=device))

    adj_tensor = torch.Tensor(adj_table)
    adj_tensor.to(device)

    mod.train()
    print(mod.state_dict())
    itr = tqdm.trange(100)
    losses = []
    for i in itr:
        for t in tqdm.trange(time_size - 1):
            Z, theta = mod(input_list[t], population_list[t])

            loss = objective(Z, theta, population_list[t], population_list[t+1], 1.0)
            losses.append(loss.item())
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #itr.set_postfix(ordered_dict=OrderedDict(loss=loss.item(), b_grad=mod.fc2.bias.grad))
            itr.set_postfix(ordered_dict=OrderedDict(b_grad=mod.fc2.bias.grad))
    print(mod.state_dict())
    print(Z)
    print(theta)