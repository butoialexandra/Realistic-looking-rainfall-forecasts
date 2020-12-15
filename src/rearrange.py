import torch
import os
import glob

base_dir = "/mnt/ds3lab-scratch/dslab2019/shghosh/preprocessed"

idx_set = range(1,37)


for idx in idx_set:
    filename = f"{base_dir}/x.{idx}.pt"
    yarr  = torch.load(filename).reshape(-1,127,188)
    torch.save(yarr, f"{base_dir}/predictions.{idx}.pt")

for idx in idx_set:
    filename = f"{base_dir}/y.{idx}.pt"
    yarr  = torch.load(filename).reshape(-1,127,188)
    torch.save(yarr, f"{base_dir}/observations.{idx}.pt")

for idx in idx_set:
    filename = f"{base_dir}/yHighres.{idx}.pt"
    y_hr_arr = torch.load(filename).reshape(-1,2*127-1, 2*188-1)
    torch.save(y_hr_arr, f"{base_dir}/observations_hr.{idx}.pt")
