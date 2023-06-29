import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSCanonical
import random
from sklearn.neighbors import KDTree
from annoy import AnnoyIndex
# from geosketch import gs
import scipy.sparse as sps
from itertools import product
from fbpca import pca

SCALE_BEFORE = True
EPS = 1e-12
VERBOSE = False


def reduce_dimensionality(X, dim_red_k=100):
    k = min((dim_red_k, X.shape[0], X.shape[1]))
    U, s, Vt = pca(X, k=k) # Automatically centers.
    return U[:, range(k)] * s[range(k)]

def svd1(mat, num_cc):
    U, s, V = np.linalg.svd(mat)
    d = s[0:int(num_cc)]
    u = U[:, 0:int(num_cc)]
    v = V[0:int(num_cc), :].transpose()
    return u, v, d

def pls(x, y, num_cc):
    random.seed(42)
    plsca = PLSCanonical(n_components=int(num_cc), algorithm='svd')
    fit = plsca.fit(x, y)
    u = fit.x_weights_
    v = fit.y_weights_
    a1 = np.matmul(np.matrix(x), np.matrix(u)).transpose()
    d = np.matmul(np.matmul(a1, np.matrix(y)), np.matrix(v))
    ds = [d[i, i] for i in range(0, 30)]
    return u, v, ds



#' @param data Input data
#' @param query Data to query against data
#' @param k Number of nearest neighbors to compute
# Approximate nearest neighbors using locality sensitive hashing.
def NN(data, query=None, k=10, metric='manhattan', n_trees=10):
    if query is None:
        query = data

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
    
