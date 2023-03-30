import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
import tables
import torch
import torch.nn as nn
import scipy.sparse as sps
import torch.nn.functional as F
import torch.optim as optim

from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

from .model import *
from .dataset import ClsDataset, hvg_binz_lognorm_scale
from .utils import objectview, adjust_learning_rate
from .loss import L1regularization, InfoNCE
from .loss import CosineHead, CosineMarginHead, SoftmaxMarginHead
from .sNNs import NN
from .knn_classifier import kNN_approx, knn_classifier_top_k, knn_classifier_eval
from .metrics import osr_evaluator

import torch.utils.data.dataloader as dataloader
from line_profiler import LineProfiler


class scNCL(object):
    def __init__(self, 
                encoder_type='linear', n_latent=20, bn=False, dr=0.2, 
                l1_w=0.1, ortho_w=0.1, 
                cont_w=0.0, cont_tau=0.4, cont_cutoff=0.,
                align_w=0.0, align_p=0.8, align_cutoff=0.,
                clamp=None,
                seed=1234
                ):

        # add device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.n_latent = n_latent
        self.encoder_type = encoder_type
        self.bn = bn
        self.dr = dr
        self.l1_w = l1_w
        self.ortho_w = ortho_w
        self.cont_w = cont_w
        self.cont_tau = cont_tau
        self.cont_cutoff = cont_cutoff
        self.align_w = align_w
        self.align_p = align_p
        self.align_cutoff = align_cutoff
        self.clamp = clamp
        

    def preprocess(self, 
                   adata_inputs,   # list of 'anndata' object
                   atac_raw_emb,   
                   pp_dict,
                   adata_adt_inputs=None, # list of adata_adt 
                   ):
        '''
        Performing preprocess for a pair of datasets.
        '''
        rna = adata_inputs[0].copy()
        atac = adata_inputs[1].copy()
        n_rna, n_atac = rna.shape[0], atac.shape[0]
        n_feature1, n_feature2 = rna.shape[1], atac.shape[1]
        assert n_feature1 == n_feature2, 'unmatched feature dim'
        assert (rna.var_names == atac.var_names).all(), 'unmatched feature names'

        self.batch_label = pp_dict['batch_label']
        self.type_label  = pp_dict['type_label']

        rna, atac, hvg_total = hvg_binz_lognorm_scale(rna, atac, pp_dict['hvg_num'], pp_dict['binz'], 
                                            pp_dict['lognorm'], pp_dict['scale_per_batch'])
        self.hvg_total = hvg_total

        self.data_A = sps.csr_matrix(rna.X)
        self.data_B = sps.csr_matrix(atac.X)
        self.emb_B = atac_raw_emb

        if adata_adt_inputs is not None:
            print('Concating adt features...')
            csr_adt_a = sps.csr_matrix(adata_adt_inputs[0].X)
            self.data_A = sps.csr_matrix(sps.hstack([self.data_A, csr_adt_a]))
              
            csr_adt_b = sps.csr_matrix(adata_adt_inputs[1].X)
            self.data_B = sps.csr_matrix(sps.hstack([self.data_B, csr_adt_b]))

        self.n_input = self.data_A.shape[1]
        self.n_rna, self.n_atac = n_rna, n_atac
        self.meta_A = rna.obs.copy()
        self.meta_B = atac.obs.copy() 

        y_A = self.meta_A[self.type_label].values
        # y_B = self.meta_B[self.type_label].values

        self.relabel(y_A)

        # self.share_mask = np.in1d(y_B, self.class_A) 
        # self.share_class_name = np.unique(y_B[self.share_mask])

        self.shuffle_data()

        # get neighborhood graph for NCL
        self.get_nns(pp_dict['knn'], pp_dict['knn_by_tissue'])

    def relabel(self, y_A):
        self.y_A = y_A

        self.class_A = np.unique(self.y_A)
        # self.class_B = np.unique(self.y_B)

        self.trainlabel2id = {v:i for i,v in enumerate(self.class_A)}
        self.id2trainlabel = {v:k for k,v in self.trainlabel2id.items()}

        self.y_id_A = np.array([self.trainlabel2id[_] for _ in self.y_A]).astype('int32')
        # self.y_id_B = np.array([self.trainlabel2id.get(_, -1) for _ in self.y_B]).astype('int32')
        self.n_class = len(self.class_A)

    def shuffle_data(self):
        # shuffle source domain
        rand_idx_ai = np.arange(self.n_rna)
        np.random.shuffle(rand_idx_ai)
        self.data_A_shuffle = self.data_A[rand_idx_ai]
        self.meta_A_shuffle = self.meta_A.iloc[rand_idx_ai]
        self.y_A_shuffle = self.y_A[rand_idx_ai]
        self.y_id_A_shuffle = self.y_id_A[rand_idx_ai].astype('int32')

        # shuffle target domain
        random_idx_B = np.arange(self.n_atac)
        np.random.shuffle(random_idx_B)
        self.data_B_shuffle = self.data_B[random_idx_B]
        self.emb_B_shuffle = self.emb_B[random_idx_B]
        self.meta_B_shuffle = self.meta_B.iloc[random_idx_B]
        # self.y_B_shuffle = self.y_B[random_idx_B]
        # self.y_id_B_shuffle = self.y_id_B[random_idx_B].astype('int32')

    def get_nns(self, k=10, knn_by_tissue=False):
        if not knn_by_tissue:
            knn_ind = NN(self.emb_B_shuffle, query=self.emb_B_shuffle, k=k+1, metric='manhattan', n_trees=10)[:, 1:]
        else:
            assert 'tissue' in self.meta_B_shuffle.columns, 'tissue not found in metadata'

            tissue_set = self.meta_B_shuffle['tissue'].unique()
            tissue_vec = self.meta_B_shuffle['tissue'].values
            knn_ind = np.zeros((self.n_atac, k))

            for ti in tissue_set:
                ti_idx = np.where(tissue_vec==ti)[0]
                ti_knn_ind = NN(self.emb_B_shuffle[ti_idx], self.emb_B_shuffle[ti_idx], k=k+1)[:, 1:]
                knn_ind[ti_idx, :] = ti_idx[ti_knn_ind.ravel()].reshape(ti_knn_ind.shape)

        knn_ind = knn_ind.astype('int64')

        if self.type_label in self.meta_B_shuffle.columns:
            y_ = self.meta_B_shuffle[self.type_label].to_numpy()
            y_knn = y_[knn_ind.ravel()].reshape(knn_ind.shape)
                
            ratio = (y_.reshape(-1, 1) == y_knn).mean(axis=1).mean()
            print('==========================')
            print('knn correct ratio = {:.4f}'.format(ratio))
            print('==========================')

        self.knn_ind = knn_ind

    def cor(self, m):   # covariance matrix of embedding features
        m = m.t()
        fact = 1.0 / (m.size(1) - 1)
        m = m - torch.mean(m, dim=1, keepdim=True)
        mt = m.t()
        return fact * m.matmul(mt).squeeze()

    def euclidean(self, x1, x2):
        return ((x1-x2)**2).sum().sqrt()

    def non_corr(self, x):
        l = torch.mean(torch.abs(torch.triu(self.cor(x), diagonal=1)))
        return l

    def zero_center(self, x):  # control value magnitude
        l = torch.mean(torch.abs(x))
        return l

    def max_var(self, x):
        l = max_moment1(x)
        return l

    def get_pos_ind(self, ind):
        choice_per_nn_ind = np.random.randint(low=0, high=self.knn_ind.shape[1], size=ind.shape[0])
        pos_ind = self.knn_ind[ind, choice_per_nn_ind]
        return pos_ind

    def init_train(self, opt, lr, lr2, weight_decay):
        # feature extractor
        if self.encoder_type == 'linear':
            self.encoder = torch.nn.DataParallel(Net_encoder(self.n_input, self.n_latent).cuda())
        else:
            self.encoder = torch.nn.DataParallel(Nonlinear_encoder(self.n_input, self.n_latent, self.bn, self.dr).cuda())

        self.head  = torch.nn.DataParallel(Net_cell(self.n_latent, self.n_class).cuda())

        if opt == 'adam':
            optimizer_G = optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=weight_decay)
            optimizer_C = optim.Adam(self.head.parameters(), lr=lr2 if lr2 is not None else lr, weight_decay=weight_decay)
        elif opt == 'sgd':
            optimizer_G = optim.SGD(self.encoder.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            optimizer_C = optim.SGD(self.head.parameters(), lr=lr2 if lr2 is not None else lr, momentum=0.9, weight_decay=weight_decay)
        return optimizer_G, optimizer_C

    # @profile
    def train_step(
            self, 
            step, batch_size, 
            optimizer_G, optimizer_C, 
            cls_crit, reg_crit, reg_cont, 
            log_step=100, 
            eval_atac=False, eval_top_k=1, eval_open=False
        ):
        self.encoder.train()
        self.head.train()

        N_A = self.n_rna
        N_B = self.n_atac

        issparse = sps.issparse(self.data_B)

        # A input
        index_A = np.random.choice(np.arange(N_A), size=batch_size)
        x_A = torch.from_numpy(self.data_A_shuffle[index_A, :].A).float().cuda()
        y_A = torch.from_numpy(self.y_id_A_shuffle[index_A]).long().cuda()

        index_B = np.random.choice(np.arange(N_B), size=batch_size)
        x_B = torch.from_numpy(self.data_B_shuffle[index_B, :].A).float().cuda()

        # forward
        f_A = self.encoder(x_A)
        if self.clamp:
            f_A = torch.clamp(f_A, min=-self.clamp, max=self.clamp)
        p_A  = self.head(f_A) 

        f_B = self.encoder(x_B)
        if self.clamp:
            f_B = torch.clamp(f_B, min=-self.clamp, max=self.clamp)
        p_B = self.head(f_B)

        optimizer_G.zero_grad()
        optimizer_C.zero_grad()

        # Adapted NNDR loss
        A_center_loss = self.zero_center(f_A)
        A_corr_loss   = self.non_corr(f_A)
        A_var_loss    = self.max_var(f_A)
    
        B_center_loss = self.zero_center(f_B)
        B_corr_loss   = self.non_corr(f_B)
        B_var_loss    = self.max_var(f_B)

        adapted_NNDR_loss = A_center_loss+B_center_loss+A_corr_loss+B_corr_loss+B_var_loss

        # NCL loss
        cont_loss = 0
        if self.cont_w != 0 and (step>=self.cont_cutoff):
            B_pos_ind = self.get_pos_ind(index_B)
            x_B_pos = torch.from_numpy(self.data_B_shuffle[B_pos_ind, :].A).float().cuda()

            f_B_pos = self.encoder(x_B_pos)
            if self.clamp:
                f_B_pos = torch.clamp(f_B_pos, min=-self.clamp, max=self.clamp)
            cont_loss = reg_cont(f_B, f_B_pos)

        # Alignment loss
        align_loss = 0.    
        if (self.align_w != 0) and (step >= self.align_cutoff):
            bs = f_B.size(0)
            # cosine similarity loss  
            f_A_norm = F.normalize(f_A, p=2, dim=1)
            f_B_norm = F.normalize(f_B, p=2, dim=1)

            f_A_norm_detach, f_B_norm_detach = f_A_norm.detach(), f_B_norm.detach()  

            cos_sim = torch.matmul(f_B_norm_detach, f_A_norm_detach.t())
            vals, inds = torch.max(cos_sim, dim=1)
            vals, top_B_inds = torch.topk(vals, int(bs * self.align_p))
            top_B_A_inds = inds[top_B_inds]  # corresponding A indices

            # maximize similarity between top_B_inds, top_B_A_inds
            f_B_norm_top = f_B_norm[top_B_inds]
            f_A_norm_top = f_A_norm[top_B_A_inds]

            align_loss = -torch.mean(torch.sum(f_A_norm_top * f_B_norm_top, dim=1))  # -cos_similarity

        # Supervised classification loss
        loss_cls = cls_crit(p_A, y_A)

        # Regularization loss
        l1_reg_loss = reg_crit(self.encoder) + reg_crit(self.head) 

        loss = loss_cls + l1_reg_loss + self.ortho_w*adapted_NNDR_loss + self.cont_w*cont_loss + self.align_w*align_loss

        loss.backward()
        optimizer_G.step()
        optimizer_C.step()

        # logging info
        if not (step % log_step):
            print("step %d, loss_cls=%.3f, loss_l1_reg=%.3f, center=(%.3f, %.3f), corr=(%.3f, %.3f), var=(%.3f, %.3f), loss_cont=%.3f, loss_align=%.3f" % \
                (step, loss_cls, l1_reg_loss, \
                 self.ortho_w*A_center_loss, self.ortho_w*B_center_loss, self.ortho_w*A_corr_loss, self.ortho_w*B_corr_loss, \
                 self.ortho_w*A_var_loss, self.ortho_w*B_var_loss, \
                 self.cont_w*cont_loss, 
                 self.align_w*align_loss
                )
            )

            feat_A, feat_B, head_A, head_B = self.eval(inplace=False)
            pr_A = np.argmax(head_A, axis=1)
            pr_B = np.argmax(head_B, axis=1)
            pr_B_top_k = np.argsort(-1 * head_B, axis=1)[:, :eval_top_k]

            # if cell type annotation of scATAC-seq data available
            # then, evaluate the performance
            if eval_atac and (self.type_label in self.meta_B.columns):
                y_B = self.meta_B[self.type_label].to_numpy()
                y_id_B = np.array([self.trainlabel2id.get(_, -1) for _ in y_B])

                share_mask = np.in1d(y_B, self.class_A)
                pr_B_top_acc = knn_classifier_eval(pr_B_top_k, y_id_B, True, share_mask)
            
                if not eval_open:  # close-set eval
                    print("Overall acc={:.5f}".format(pr_B_top_acc))
                else:              # open-set eval
                    closed_score = np.max(head_B, axis=1)
                    open_score   = 1 - closed_score

                    kn_data_pr = pr_B[share_mask]
                    kn_data_gt = y_id_B[share_mask]
                    kn_data_open_score = open_score[share_mask]
                    unk_data_open_score = open_score[np.logical_not(share_mask)]

                    closed_acc, os_auroc, os_aupr, oscr = osr_evaluator(kn_data_pr, kn_data_gt, kn_data_open_score, unk_data_open_score)

        return loss_cls.item()


    def train(self, 
            opt='sgd', 
            batch_size=500, training_steps=2000, 
            lr=0.001, lr2=None, weight_decay=5e-4,
            log_step=100, eval_atac=False, eval_top_k=1, eval_open=False,
            ):
        # torch.manual_seed(1)
        begin_time = time.time()    

        # init model
        optimizer_G, optimizer_C = self.init_train(opt, lr, lr2, weight_decay)

        reg_crit = L1regularization(self.l1_w).cuda()
        reg_cont = InfoNCE(batch_size, self.cont_tau).cuda()
        # cls_crit = nn.CrossEntropyLoss(reduction='none').cuda()
        cls_crit = nn.CrossEntropyLoss().cuda()

        self.loss_cls_history = []
        for step in range(training_steps):
            loss_cls = self.train_step( 
                step, batch_size,
                optimizer_G=optimizer_G, optimizer_C=optimizer_C, 
                cls_crit=cls_crit, reg_crit=reg_crit, reg_cont=reg_cont, 
                log_step=log_step, 
                eval_atac=eval_atac, eval_top_k=eval_top_k, eval_open=eval_open
            )

            self.loss_cls_history.append(loss_cls)

        end_time = time.time()
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.train_time = end_time - begin_time
        print("Training takes %.2f seconds" % self.train_time)

    def eval(self, batch_size=500, inplace=False):
        # test loader
        src_ds = ClsDataset(self.data_A, self.y_id_A, binz=False, train=False)   # for evaluation
        tgt_ds = ClsDataset(self.data_B, np.ones(self.n_atac, dtype='int32'), binz=False, train=False)
        self.src_dl = dataloader.DataLoader(src_ds, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=False) 
        self.tgt_dl = dataloader.DataLoader(tgt_ds, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=False)

        self.encoder.eval()
        self.head.eval()

        feat_A, head_A = [], []
        for x, y in self.src_dl:
            x = x.cuda()
            z_A = self.encoder(x)
            if self.clamp:
                z_A = torch.clamp(z_A, min=-self.clamp, max=self.clamp)
            h_A = nn.Softmax(dim=1)(self.head(z_A))
            feat_A.append(z_A.detach().cpu().numpy())
            head_A.append(h_A.detach().cpu().numpy())

        feat_B, head_B = [], []
        for x, y in self.tgt_dl:
            x = x.cuda()
            z_B = self.encoder(x)
            if self.clamp:
                z_B = torch.clamp(z_B, min=-self.clamp, max=self.clamp)
            h_B = nn.Softmax(dim=1)(self.head(z_B))
            feat_B.append(z_B.detach().cpu().numpy())
            head_B.append(h_B.detach().cpu().numpy())

        feat_A, feat_B = np.vstack(feat_A), np.vstack(feat_B)
        head_A, head_B = np.vstack(head_A), np.vstack(head_B)

        if inplace:
            self.feat_A = feat_A
            self.feat_B = feat_B
            self.head_A = head_A
            self.head_B = head_B

            self.feat_AB = np.vstack([feat_A, feat_B])
            self.head_AB = np.vstack([head_A, head_B])
        else:
            return feat_A, feat_B, head_A, head_B

    def load_ckpt(self, path):
        self.encoder.load_state_dict(torch.load(path)['encoder'])
        self.head.load_state_dict(torch.load(path)['head'])
        print(f'loaded checkpoints from {path}')

    def annotate(self, label_prop=False, prop_knn=10):
        try:
            self.head_B
        except:
            self.eval(inplace=True)

        atac_pr = np.argmax(self.head_B, axis=1)
        if label_prop:
            atac_pr = kNN_approx(self.feat_B, self.feat_B, atac_pr, n_sample=None, knn=prop_knn)

        atac_pr = np.array([self.id2trainlabel[_] for _ in atac_pr])
        return atac_pr


def max_moment0(feats):
    loss = 1 / torch.mean(torch.abs(feats - torch.mean(feats, dim=0)))
    return loss

def max_moment1(feats):
    loss = 1 / torch.mean(   
            torch.abs(feats - torch.mean(feats, dim=0)))
    return loss

def inter_class_dist(v_cls, feats):
    cls_set = np.unique(v_cls)
    cls_centers = []
    for i, ci in enumerate(cls_set):
        ci_mask = v_cls == ci
        cls_centers.append(feats[ci_mask].mean(axis=0))
    cls_centers = np.vstack(cls_centers)

    inter_dist = pairwise_distances(cls_centers)
    ds = np.tril_indices(inter_dist.shape[0], k=-1)  # below the diagonal
    v = inter_dist[ds].mean()
    return v

def intra_class_dist(v_cls, feats):
    cls_set = np.unique(v_cls)
    cls_vars = []
    for i, ci in enumerate(cls_set):
        ci_mask = v_cls == ci
        cent = feats[ci_mask].mean(axis=0, keepdims=True)
        ci_var = pairwise_distances(cent, feats[ci_mask]).mean()
        cls_vars.append(ci_var)
    intra_var = np.mean(cls_vars)
    return intra_var

def measure_var(v_cls, feats, probs, l2norm=False):
    if l2norm:
        feats = normalize(feats, axis=1)  

    inter_var = inter_class_dist(v_cls, feats)
    intra_var = intra_class_dist(v_cls, feats)
    total_var = inter_var / intra_var
    return inter_var, intra_var, total_var

def save_ckpts(output_dir, model, step):
    state = {
            'encoder': model.encoder.state_dict(), 
            'head': model.head.state_dict(), 
            }
    torch.save(state, os.path.join(output_dir, f"ckpt_{step}.pth"))


