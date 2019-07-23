import torch
import tqdm

class Data_loader():
    def __init__(self, population_data, location, adj_table, neighbor, time_size, location_size, neighbor_size, device):
        self.input_list = []
        self.population_list = []
        self.adj_tensor = torch.Tensor(adj_table)
        self.neighbor_table = neighbor
        self.device = device
        
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
            self.input_list.append(torch.tensor(input_list_tmp, dtype=torch.double))
            self.population_list.append(torch.tensor(population_data[t], dtype=torch.double))

    
    def get_t_input(self, t):
        return self.input_list[t].to(self.device), self.population_list[t].to(self.device), self.population_list[t + 1].to(self.device)