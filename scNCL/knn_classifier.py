import os
import os.path
import sys
import pdb

import numpy as np
import time

try:
    import faiss
    from faiss import normalize_L2
except:
    pass
from annoy import AnnoyIndex

import scipy
import torch.nn.functional as F
import torch
from functools import partial

import scipy.stats as stats
from sklearn.preprocessing import normalize

def annoy_knn(data, query=None, k=10, metric='manhattan', n_trees=10):
    if query is None:
        query = data

    data = normalize(data, axis=1)
    query = normalize(query, axis=1)

    # Build index.
    a = AnnoyIndex(data.shape[1], metric=metric)
    for i in range(data.shape[0]):
        a.add_item(i, data[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(query.shape[0]):
        ind.append(a.get_nns_by_vector(query[i, :], k, search_k=-1))
    ind = np.array(ind)

    return ind


def faiss_knn(X1, X2, knn, return_dist=False):
    d = X1.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatIP(res,d,flat_config)   # build the index

    normalize_L2(X1)
    normalize_L2(X2)
    index.add(X1) 
    N = X1.shape[0]

    c = time.time()
    D, I = index.search(X2, knn)
    elapsed = time.time() - c
    print('kNN Search done in %d seconds' % elapsed)

    # I = I[:, 1:]

    if return_dist:
        return I, D
    else:
        return I

def equal_inter_sampling(X, Y, n_sample):
    n_total = X.shape[0]
    if n_sample < n_total:
        smp_ind = np.linspace(0, n_total-1, num=n_sample).astype('int')
        X = X[smp_ind]
        Y = Y[smp_ind]
    return X, Y

def kNN_approx(X1, X2, Y1, n_sample=20000, knn=10, knn_method='annoy'):
    X1 = X1.copy()
    X2 = X2.copy()
    Y1 = Y1.copy()

    if n_sample is not None:
        X1, Y1 = equal_inter_sampling(X1, Y1, n_sample=n_sample)

    if knn_method == 'faiss':
        knn_ind = faiss_knn(X1, X2, knn)
    else:
        knn_ind = annoy_knn(X1, X2, knn)

    knn_pred_pop = Y1[knn_ind.ravel()].reshape(knn_ind.shape)

    knn_pred = stats.mode(knn_pred_pop, axis=1)[0].ravel()
    return knn_pred

def find_top_k(knn_pred, top_k=2):
    clas_vote, n_vote = np.unique(knn_pred, return_counts=True)
    sort_ind = np.argsort(-1 * n_vote)
    clas_vote_sort = clas_vote[sort_ind]

    return clas_vote_sort[:top_k]

def knn_classifier_top_k(X1, X2, Y1, n_sample=20000, knn=10, top_k=10, knn_method='annoy'):
    top_k = min(top_k, knn)
    X1 = X1.copy()
    X2 = X2.copy()
    Y1 = Y1.copy()

    if n_sample is not None:
        X1, Y1 = equal_inter_sampling(X1, Y1, n_sample=n_sample)

    if knn_method == 'faiss':
        knn_ind = faiss_knn(X1, X2, knn)
    else:
        knn_ind = annoy_knn(X1, X2, knn)
    knn_pred_pop = Y1[knn_ind.ravel()].reshape(knn_ind.shape)

    knn_pred_top_k = map(partial(find_top_k, top_k=top_k), knn_pred_pop)
    knn_pred_top_k = list(knn_pred_top_k)

    return knn_pred_top_k

def knn_classifier_eval(knn_pr, y_gt, top_k=False, mask=None):    
    corr_mask = []
    for x,y in zip(knn_pr, y_gt):
        if top_k:
            corr_mask.append(y in x)
        else:
            corr_mask.append(x==y)
    corr_mask = np.array(corr_mask)
    
    mask = np.ones(len(y_gt)).astype('bool') if mask is None else mask
    acc = corr_mask[mask].mean()
    # print(acc)
    return acc


# def knn_classifier_prob(X1, X2, Y1, n_sample=None, knn=10, crit='count'):
#     '''
#     return :
#         target_neighbor: predicted label
#         traget_prob: confidence score
#     '''
#     n, dim = X1.shape[0], X1.shape[1]
#     X1 = X1.copy()
#     X2 = X2.copy()
#     Y1 = Y1.copy()

#     if n_sample is not None:
#         X1, Y1 = equal_inter_sampling(X1, Y1, n_sample=n_sample)

#     knn_ind, knn_simil = faiss_knn(X1, X2, knn, return_dist=True)
#     knn_pred_pop = Y1[knn_ind.ravel()].reshape(knn_ind.shape)

#     def prob_f(simi, labs, crit='simi'):
#         sort_inds = np.argsort(labs)
#         labs = labs[sort_inds]
#         simi = simi[sort_inds]
#         values, inds, counts = np.unique(labs, return_index=True, return_counts=True)
#         simi_list = np.split(simi, inds[1:])
#         simi_accum = list(map(lambda x: x.sum(), simi_list))

#         if crit == 'simi':
#             iid = np.argmax(simi_accum)
#             lab = values[iid]
#         elif crit == 'count':
#             iid = np.argmax(counts)
#             lab = values[iid]
#         prob = simi_accum[iid] / counts[iid]

#         return lab, prob

#     # knn_simil = 1 - knn_dist
#     res = list(map(partial(prob_f, crit=crit), knn_simil, knn_pred_pop))

#     return res  

# Implementations from concerto
# https://github.com/melobio/Concerto-reproducibility/blob/ab1fc7f86823740ee3494703b8963cf2bd06e45f/concerto_function5_3.py#L2193
def knn_classifier_prob_concerto(X1, X2, Y1, n_sample=None, knn=10, num_chunks=100):
    '''
    return :
        target_neighbor: predicted label
        traget_prob: confidence score
    '''
    X1 = X1.copy()
    X2 = X2.copy()
    Y1 = Y1.copy()

    num_test_images = int(X2.shape[0])
    imgs_per_chunk = num_test_images // num_chunks
    if imgs_per_chunk == 0:
        imgs_per_chunk = 10

    if n_sample is not None:
        X1, Y1 = equal_inter_sampling(X1, Y1, n_sample=n_sample)

    X1 = normalize(X1, axis=1)
    X2 = normalize(X2, axis=1)

    target_pred_labels = []
    target_pred_prob = []
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = X2[
                   idx: min((idx + imgs_per_chunk), num_test_images), :
                   ]
        similarity = features.dot(X1.T)
        partition_indices = np.argpartition(-similarity, knn, axis=1)
        target_similarity = np.take_along_axis(similarity, partition_indices[:, :knn], axis=1)
        target_indices    = partition_indices[:, :knn]

        # target_distances, target_indices = tf.math.top_k(similarity, k, sorted=True)

        for simil, indices in zip(target_similarity, target_indices):   
            selected_label = {}
            selected_count = {}
            count = 0
            for sim, index in zip(simil, indices):
                label = Y1[index]
                weight = sim
                if label not in selected_label:
                    selected_label[label] = 0.
                    selected_count[label] = 0
                selected_label[label] += weight
                selected_count[label] += 1
                count += 1

            filter_label_list = sorted(selected_label.items(), key=lambda x: x[1], reverse=True) 
            target_pred_labels.append(filter_label_list[0][0])

            prob = selected_label[filter_label_list[0][0]] / selected_count[filter_label_list[0][0]]  
            target_pred_prob.append(prob)

    target_pred = np.array(target_pred_labels)
    target_prob = np.array(target_pred_prob)

    return target_pred, target_prob 




