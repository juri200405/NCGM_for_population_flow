import torch
import tqdm

class Data_loader():
    def __init__(self, population_data, location, adj_table, neighbor, time_size, location_size, neighbor_size, device):
        self.adj_tensor = torch.Tensor(adj_table)
        self.population_tensor = torch.tensor(population_data, dtype=torch.double)
        self.neighbor_table = neighbor
        self.device = device
        
        input_list = []
        for t in tqdm.trange(time_size - 1):
            input_list_tmp = []
            for l in range(location_size):
                input_tmp = []
                #for ll in neighbor_table[l]:
                for ll in range(location_size):
                    input_tmp.append([t / float(time_size) - 0.5, location[l][0], location[l][1], location[ll][0] - location[l][0], location[ll][1] - location[l][1]])
                while len(input_tmp) < neighbor_size:
                    input_tmp.append([t / float(time_size) - 0.5, location[l][0], location[l][1], 0.0, 0.0])
                input_list_tmp.append(input_tmp)
            input_list.append(input_list_tmp)
        self.input = torch.tensor(input_list, dtype=torch.double)

    
    def get_t_input(self, t):
        return self.input.to(self.device), self.population_tensor.to(self.device)