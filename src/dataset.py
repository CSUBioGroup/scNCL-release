import torch
import random
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sps
from torch.utils.data import Dataset

random.seed(1)

def hvg_binz_lognorm_scale(rna, atac, hvg_num, binz, lognorm, scale_per_batch):
    n_rna, n_atac = rna.shape[0], atac.shape[0]
    n_feature1, n_feature2 = rna.shape[1], atac.shape[1]

    print("Finding highly variable genes...")
    if hvg_num <= n_feature1:
        sc.pp.highly_variable_genes(rna, flavor='seurat_v3', n_top_genes=hvg_num)
        sc.pp.highly_variable_genes(atac, flavor='seurat_v3', n_top_genes=hvg_num)

        hvg_rna  = rna.var[rna.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        hvg_atac = atac.var[atac.var.highly_variable == True].sort_values(by="highly_variable_rank").index
        hvg_total = (hvg_rna) & (hvg_atac)

        if len(hvg_total) < 100:
            raise ValueError("The total number of highly variable genes is smaller than 100 (%d). Try to set a larger hvg_num." % len(hvg_total))
    else:
        hvg_total = rna.var_names

    hvg_total = list(hvg_total[:hvg_num])

    ## pp
    if binz:
        rna.X = (rna.X>0).astype('float')
        atac.X = (atac.X>0).astype('float')
    if lognorm:
        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)

        sc.pp.normalize_total(atac, target_sum=1e4)
        sc.pp.log1p(atac)

    # subset genes    
    rna  = rna[:, hvg_total].copy()
    atac = atac[:, hvg_total].copy()

    if scale_per_batch:
        sc.pp.scale(rna, max_value=10)
        sc.pp.scale(atac, max_value=10)

    return rna, atac, hvg_total

class ClsDataset(Dataset):
    def __init__(
            self, 
            feats, labels, binz=True, train=False, return_id=False
        ):
        self.X = feats
        self.y = labels
        self.train = train
        self.binz = binz
        self.return_id = return_id
        self.sample_num = self.X.shape[0]
        self.input_size = self.X.shape[1]
        self.issparse = sps.issparse(feats)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if self.train:        # the same as scjoint, but never used
            rand_idx = random.randint(0, self.sample_num - 1)
            sample = self.X[rand_idx].A if self.issparse else self.X[rand_idx]
            sample = sample.reshape((1, self.input_size))
            in_data = (sample>0).astype('float32') if self.binz else sample.astype('float32')  # binarize data
            in_label = self.y[rand_idx]
            i = rand_idx
        else:
            sample = self.X[i].A if self.issparse else self.X[i]
            in_data = (sample>0).astype('float32') if self.binz else sample.astype('float32')
            in_label = self.y[i]

        # x = self.data[i].A         # if binarize_data, use this
        if self.return_id:
            return in_data.squeeze(), in_label, i
        else:
            return in_data.squeeze(), in_label


class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B, max_dataset_size):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        A_x, A_y = None, None
        B_x, B_y = None, None
        try:
            A_x, A_y = next(self.data_loader_A_iter)
        except StopIteration:
            if A_x is None or A_y is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A_x, A_y = next(self.data_loader_A_iter)

        try:
            B_x, B_y = next(self.data_loader_B_iter)
        except StopIteration:
            if B_x is None or B_y is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B_x, B_y = next(self.data_loader_B_iter)

        if (self.stop_A and self.stop_B) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'A_x': A_x, 'A_y': A_y,
                    'B_x': B_x, 'B_y': B_y}


