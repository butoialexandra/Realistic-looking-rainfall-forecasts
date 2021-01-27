from verification import *
from dataset import ConditionalDataset
import socket
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False, default='cosmo', help='cosmo | cond-cosmo')
    parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
    parser.add_argument('--device', type=int, default=0, help='selects cuda device')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    opt = parser.parse_args()


    on_cluster = socket.gethostname().endswith(".ethz.ch")
    print(f"Running on cluster: {on_cluster}")
    device = torch.device(f"cuda:{opt.device}" if opt.cuda else "cpu")

    dataset = ConditionalDataset(device=device, highres=True)

    num_data = len(dataset)
    test_idx = range(math.ceil(0.9 * num_data), num_data)
    valid_ds = torch.utils.data.Subset(dataset, test_idx)
    print("Valid dataset length: {}".format(len(valid_ds)))
    valid_dataloader = torch.utils.data.DataLoader(valid_ds, batch_size=opt.batchSize, shuffle=False,
                                                   num_workers=int(opt.workers))

    lsd = 0
    rmse = 0
    w_ratio = 0
    step = 0

    rescale = torchvision.transforms.Resize(size=[128,196])
    for i, (x, y) in enumerate(valid_ds, 0):
        step += 1
        data = y
        real = data.to(device)
        pred = x.to(device)
        real = real.float()
        pred = pred.float()

        real = rescale(real)
        real = real.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        lsd += log_spectral_distance_pairs_avg(real, pred)
        rmse += root_mse(pred, real)
        w_ratio += wetness_ratio(pred, real)

    lsd, rmse, w_ratio = lsd / step, rmse / step, w_ratio / step
    print("Log spectral distance: {}".format(lsd))
    print("Root mean squared error: {}".format(rmse))
    print("Wetness ratio: {}".format(w_ratio))
