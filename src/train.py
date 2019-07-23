from pathlib import Path
import tqdm
from collections import OrderedDict

import torch
import torch.optim as optim

import tensorboardX

import model
import datas
import dataloader

if __name__ == "__main__":
    is_samlpe = True
    if is_samlpe:
        neighbor_size = 4
        time_size = 8
        location_size = 4
        population_data, location, adj_table, neighbor_table = datas.read_samlpe()
    else:
        neighbor_size = 10
        time_size = 9480
        location_size = 117
        population_data, adj_table, location_table, neighbor_table = datas.read_data(Path("datas/chohu"), "chohu_01.csv", False)
        location = [[row[0] / 11 - 0.5, row[1] / 14 - 0.5] for row in location_table]

    use_cuda = True
    available_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if (use_cuda and available_cuda) else 'cpu')
    print(device)
    torch.set_default_dtype(torch.double)
    #torch.set_grad_enabled(True)
    #torch.autograd.set_detect_anomaly(True)

    writer = tensorboardX.SummaryWriter("log")

    mod = model.NCGM(5, 40, location_size, neighbor_size)
    mod.to(device)

    objective = model.NCGM_objective(location_size, neighbor_size, device)
    #optimizer = optim.SGD(mod.parameters(), lr=0.5)
    optimizer = optim.Adam(mod.parameters(), lr=0.1)

    data_loader = dataloader.Data_loader(population_data, location, adj_table, neighbor_table, time_size, location_size, neighbor_size, device)

    mod.train()
    itr = tqdm.trange(10000)
    losses = []
    for i in itr:
        for t in tqdm.trange(time_size - 1):
            input_data, yt, yt1 = data_loader.get_t_input(t)

            theta = mod(input_data)

            loss, Z = objective(theta, yt, yt1, 1.0)
            #print(loss)
            losses.append(loss.item())
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            itr.set_postfix(ordered_dict=OrderedDict(loss=loss.item(), b_grad=mod.fc2.bias.grad))
            #itr.set_postfix(ordered_dict=OrderedDict(loss=loss.item()))

            writer.add_scalar("loss", loss.item(), i * (time_size - 1) + t)
            writer.add_text("Z", str(Z), i * 10000 + t)
    print(theta)
    writer.close()