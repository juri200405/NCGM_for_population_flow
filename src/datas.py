import csv
from pathlib import Path

def read_data(dpath, filename, make_adj_list):
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
    
    if make_adj_list:
        build_adj_list(adj)

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

def build_adj_list(adj_table):
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