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
    #サンプルデータ（動作確認用の極小規模データ）を使うかどうか
    is_samlpe = True

    #各値の設定
    if is_samlpe:
        neighbor_size = 4
        time_size = 8
        location_size = 4
        population_data, location, adj_table, neighbor_table = datas.read_samlpe()
    else:
        #neighbor_size = 10
        neighbor_size = 117
        time_size = 9480
        location_size = 117
        population_data, adj_table, location_table, neighbor_table = datas.read_data(Path("datas/chohu"), "chohu_01.csv", False)
        location = [[row[0] / 11 - 0.5, row[1] / 14 - 0.5] for row in location_table]

    #cudaを使うかどうかの設定
    use_cuda = True
    available_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if (use_cuda and available_cuda) else 'cpu')
    print(device)

    #テンソルのデータ型のデフォルトを変更
    torch.set_default_dtype(torch.double)
    #torch.set_grad_enabled(True)
    #torch.autograd.set_detect_anomaly(True)

    #可視化ツール(tensorboardX)の設定
    writer = tensorboardX.SummaryWriter("log")

    #モデルのインスタンス化
    mod = model.NCGM(5, 40, time_size, location_size, neighbor_size)
    mod.to(device)

    #目的関数のインスタンス化
    objective = model.NCGM_objective(location_size, neighbor_size)

    #最適化関数のインスタンス化
    #optimizer = optim.SGD(mod.parameters(), lr=0.5)
    optimizer = optim.Adam(mod.parameters())

    #データ読み込みクラスのインスタンス化
    data_loader = dataloader.Data_loader(population_data, location, adj_table, neighbor_table, time_size, location_size, neighbor_size, device)

    #学習
    mod.train()
    itr = tqdm.trange(10000)
    losses = []
    ave_loss = 0.0
    for i in itr:
        for t in tqdm.trange(time_size - 1):
            #input_data : 各セルごとに5次元
            #yt : そのタイムステップでの各セルの人口
            #yt1 : 次のタイムステップでの各セルの人口
            input_data, yt, yt1 = data_loader.get_t_input(t)

            theta = mod(input_data)

            loss = objective(theta, mod.Z[t], yt, yt1, 1.0)
            #print(loss)
            losses.append(loss.item())
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #itr.set_postfix(ordered_dict=OrderedDict(loss=loss.item(), b_grad=mod.fc2.bias.grad))
            itr.set_postfix(ordered_dict=OrderedDict(loss=loss.item()))

            writer.add_scalar("loss", loss.item(), i * (time_size - 1) + t)
            ave_loss = ave_loss + loss.item()
            
        writer.add_text("Z", str(mod.Z), i)
        writer.add_scalar("ave_loss", ave_loss / (time_size - 1), i)
        ave_loss = 0.0
        with open("output/{0:05}.txt".format(i), 'wt') as f:
            f.write(str(mod.Z.data.tolist()))
    print(theta)
    writer.add_text("progress", "finish", 0)
    writer.close()
