import pickle as pkl
import scipy.sparse
import scipy.sparse as sps
import numpy as np
import pandas as pd
from scipy import sparse as sp
import networkx as nx
from data import *
from collections import defaultdict
from scipy.stats import uniform
import tensorflow as tf


#' -------- convert graph to specific format -----------

def get_value(diction, specific):
    for key, val in diction.items():
        if val == specific:
            return (key)


# 这脑残代码不知道干嘛，你先用dense存储所有细胞的MNNgraph，再转成邻接表是个什么东西？
def graph(matrix):
    adj = defaultdict(list)  # default value of int is 0，你他妈defaultdict不是个list吗？
    for i, row in enumerate(matrix):
        # cids = np.where(row)[0]
        # for j in cids:
        #     adj[i].append(j)
        for j, adjacent in enumerate(row):
            if adjacent:
                adj[i].append(j)
        if adj[i].__len__ == 0:
            adj[i] = []
    return adj

# def graph(matrix):
#     adj = defaultdict(list)

#     ridx, cidx = np.where(matrix)
#     for r, c in zip(ridx, cidx):
#         adj[r].append(c)
#     for i in range(len(matrix)):
#         if adj[i].__len__ == 0:
#             adj[i] = []
#     return adj

def create_graph(adj_ind, n):
    adj = defaultdict(list)

    for r,c in adj_ind:
        adj[r].append(c)

    for i in range(n):
        if adj[i].__len__ == 0:
            adj[i] = []
    return adj


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

# convert nested lists to a flat list
output = []
def removNestings(l):
    for i in l:
        if type(i) == list:
            removNestings(i)
        else:
            output.append(i)
    return (output)

def load_data(datadir,rgraph=True):
    input_data(datadir,Rgraph=rgraph)
    PIK = "{}/datasets.dat".format(datadir)
    with open(PIK, "rb") as f:
        objects = pkl.load(f)

    data_train1, data_test1, data_val1, label_train1, label_test1, label_val1, lab_data2, lab_label2, types = tuple(
        objects)

    train2 = pd.concat([data_train1, lab_data2])
    lab_train2 = pd.concat([label_train1, lab_label2])

    datas_train = np.array(train2)
    datas_test = np.array(data_test1)
    datas_val = np.array(data_val1)

    index_guide = np.concatenate(
        (label_train1.index, lab_label2.index * (-1) - 1, label_val1.index,
         label_test1.index))

    labels_train = np.array(lab_train2).flatten()
    labels_test = np.array(label_test1).flatten()
    labels_val = np.array(label_val1).flatten()

    #' convert pandas data frame to csr_matrix format
    datas_tr = scipy.sparse.csr_matrix(datas_train.astype('Float64'))
    datas_va = scipy.sparse.csr_matrix(datas_val.astype('Float64'))
    datas_te = scipy.sparse.csr_matrix(datas_test.astype('Float64'))
    sp.save_npz('/home/yxh/gitrepo/multi-omics/scGCN/scGCN/input/MCAsubset/tr.npz', datas_tr)
    sp.save_npz('/home/yxh/gitrepo/multi-omics/scGCN/scGCN/input/MCAsubset/va.npz', datas_va)
    sp.save_npz('/home/yxh/gitrepo/multi-omics/scGCN/scGCN/input/MCAsubset/te.npz', datas_te)

    print('tr.shape={}, va.shape={}, te.shape={}'.format(datas_tr.shape, datas_va.shape, datas_te.shape))

    #' 3) set the unlabeled data in training set

    #' @param N; the number of labeled samples in training set
    M = len(data_train1)

    #' 4) get the feature object by combining training, test, valiation sets

    features = sp.vstack((sp.vstack((datas_tr, datas_va)), datas_te)).tolil()
    features = preprocess_features(features)    # features = (coords, values, shape)

    #' 5) Given cell type, generate three sets of labels with the same dimension
    labels_tr = labels_train.flatten()
    labels_va = labels_val.flatten()
    labels_te = labels_test.flatten()

    labels = np.concatenate(
        [np.concatenate([labels_tr, labels_va]), labels_te])
    Labels = pd.DataFrame(labels)

    true_label = Labels
    #' convert list to binary matrix
    # uniq = np.unique(labels)
    n_source_class = np.unique(types).size

    rename = {v:i for i, v in enumerate(types)}  # types == reference_type_set
    indices =  np.array([rename.get(_, -1) for _ in labels])
    indices_col = np.maximum(indices, 0).astype('int64')

    # 这里的types是从reference data里来的，所以要是query里存在unknown type，这里会出现一个问题：
    # 当query label出现unseen types时，会出现这些type不被编码，即保留原始名字，则在下面创建indices时，报错
    # rename = {}
    # for line in range(0, len(types)):   
    #     key = types[line]
    #     rename[key] = int(line)

    # Label1 = Labels.replace(rename)
    # indices = np.array(Label1.values, dtype='int').tolist()

    # indice = [item for sublist in indices for item in sublist]

    #' convert list to binary matrix
    # indptr = range(len(indice) + 1)
    # dat = np.ones(len(indice))
    # binary_label = scipy.sparse.csr_matrix((dat, indice, indptr))  

    #' new label with binary values
    # 到------，这段代码给我人看傻了，能简略点吗？
    n_samples = len(indices)
    indices_row = np.arange(n_samples).astype('int64')
    new_label = sps.csr_matrix((indices>=0, (indices_row, indices_col)), 
                                shape=(n_samples, n_source_class),
                                dtype='int32').todense()

    # new_label = np.array(binary_label.todense())  # 我麻了
    idx_train = range(M)                          # 这里的M是ref data划分后的train样本数
    idx_pred = range(M, len(labels_tr))           # 这里的len(label_tr)是concat所有query后的训练样本数
    idx_val = range(len(labels_tr), len(labels_tr) + len(labels_va))
    idx_test = range(
        len(labels_tr) + len(labels_va),
        len(labels_tr) + len(labels_va) + len(labels_te))

    train_mask = sample_mask(idx_train, new_label.shape[0])
    pred_mask = sample_mask(idx_pred, new_label.shape[0])
    val_mask = sample_mask(idx_val, new_label.shape[0])
    test_mask = sample_mask(idx_test, new_label.shape[0])

    labels_binary_train = np.zeros(new_label.shape)
    labels_binary_val = np.zeros(new_label.shape)
    labels_binary_test = np.zeros(new_label.shape)
    labels_binary_train[train_mask, :] = new_label[train_mask, :]
    labels_binary_val[val_mask, :] = new_label[val_mask, :]
    labels_binary_test[test_mask, :] = new_label[test_mask, :]

    #' ----- construct adjacent matrix ---------

    id_graph1 = pd.read_csv('{}/inter_graph.csv'.format(datadir),
                            index_col=0,
                            sep=',')
    id_graph2 = pd.read_csv('{}/intra_graph.csv'.format(datadir),
                            sep=',',
                            index_col=0)

    #' --- map index ----
    fake1 = np.array([-1] * len(lab_data2.index))
    index1 = np.concatenate((data_train1.index, fake1, data_val1.index,  # 为啥这么喜欢concatenate啊，我尼玛
                             data_test1.index)).flatten()
    #' (feature_data.index==index1).all()
    fake2 = np.array([-1] * len(data_train1))
    fake3 = np.array([-1] * (len(data_val1) + len(data_test1)))   
    find1 = np.concatenate((fake2, np.array(lab_data2.index), fake3)).flatten()  # 就尼玛只会concat是吧

    #' ---------------------------------------------
    #'  intra-graph
    #' ---------------------------------------------
    id_grp1 = np.array([
        np.concatenate((np.where(find1 == id_graph2.iloc[i, 1])[0],
                        np.where(find1 == id_graph2.iloc[i, 0])[0]))
        for i in range(len(id_graph2))
    ])

    # id_grp2 = np.array([
    #     np.concatenate((np.where(find1 == id_graph2.iloc[i, 0])[0],
    #                     np.where(find1 == id_graph2.iloc[i, 1])[0]))
    #     for i in range(len(id_graph2))
    # ])
    id_grp2 = id_grp1[:, [1, 0]]  # swap row and col

    #' ---------------------------------------------
    #'  inter-graph
    #' ---------------------------------------------
    id_gp1 = np.array([
        np.concatenate((np.where(find1 == id_graph1.iloc[i, 1])[0],
                        np.where(index1 == id_graph1.iloc[i, 0])[0]))
        for i in range(len(id_graph1))
    ])

    # id_gp2 = np.array([
    #     np.concatenate((np.where(index1 == id_graph1.iloc[i, 0])[0],
    #                     np.where(find1 == id_graph1.iloc[i, 1])[0]))
    #     for i in range(len(id_graph1))
    # ])
    id_gp2 = id_gp1[:, [1, 0]]


    # matrix = np.identity(len(labels))
    # matrix[tuple(id_grp1.T)] = 1
    # matrix[tuple(id_grp2.T)] = 1
    # matrix[tuple(id_gp1.T)] = 1
    # matrix[tuple(id_gp2.T)] = 1  

    print('Constructing adjaceny graph')   # so fucking slow
    # adj = graph(matrix)
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj))
    n_samples = len(labels)
    diag_ind = np.vstack(np.diag_indices(n_samples)).T
    adj_ind = np.vstack([id_grp1, id_grp2, id_gp1, id_gp2, diag_ind])
    n_nonz = len(adj_ind)
    # adj = create_graph(adj_ind, n_samples)
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(adj))
    # adj = ((adj + adj.T) > 0).astype('int64')
    adj = sps.csr_matrix((np.ones(n_nonz), (adj_ind[:, 0], adj_ind[:, 1])), shape=(n_samples, n_samples))

    print("assign input coordinatly....")
    return adj, features, labels_binary_train, labels_binary_val, labels_binary_test, train_mask, pred_mask, val_mask, test_mask, new_label, true_label, index_guide


# feature normalization
def preprocess_features(features):   # normalize to [0,1], minmax-scaler
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


#' in case FLAGS are defined twice
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape,
                                minval=-scale,
                                maxval=scale,
                                dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape,
                                minval=-init_range,
                                maxval=init_range,
                                dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for scGCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()   # transpose挺有意思的

def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update(
        {placeholders['support'][i]: support[i]
         for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict
