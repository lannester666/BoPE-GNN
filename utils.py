import os
from os import path
import pickle as pkl
import sys
import time 

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy
import scipy.io
import scipy.sparse
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize as sk_normalize
import torch
import torch.nn.functional as F
from collections import defaultdict
import torch.nn as nn
import random
import copy 
import matplotlib.pyplot as plt
if torch.cuda.is_available():
    import scipy.io
    from torch_geometric.utils import add_self_loops, to_undirected
    from torch_sparse import SparseTensor

device = f"cuda:0" if torch.cuda.is_available() else "cpu"
EPS = 1e-10


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_test(output, labels):
    output = F.softmax(output, dim=1)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels), torch.mean(output[np.arange(output.shape[0]), preds])

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# Read split data
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data_new(dataset_str, split,normfeat):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    # print('dataset_str', dataset_str)
    # print('split', split)
    if dataset_str in ['citeseer', 'cora', 'pubmed']:
        pass
    elif dataset_str in ['chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:
        graph_adjacency_list_file_path = os.path.join(
            'new_data', dataset_str, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_str,
                                                                f'out1_node_feature_label.txt')
        graph_dict = defaultdict(list)
        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                graph_dict[int(line[0])].append(int(line[1]))
                graph_dict[int(line[1])].append(int(line[0]))

        graph_dict_ordered = defaultdict(list)
        for key in sorted(graph_dict):
            graph_dict_ordered[key] = graph_dict[key]
            graph_dict_ordered[key].sort()

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict_ordered))

        graph_node_features_dict = {}
        graph_labels_dict = {}
        if dataset_str == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(
                        line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(
                        line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        features_list = []
        for key in sorted(graph_node_features_dict):
            features_list.append(graph_node_features_dict[key])
        features = np.vstack(features_list)
        features = sp.csr_matrix(features)

        labels_list = []
        for key in sorted(graph_labels_dict):
            labels_list.append(graph_labels_dict[key])

        label_classes = max(labels_list) + 1
        labels = np.eye(label_classes)[labels_list]

        splits_file_path = 'splits/' + dataset_str + \
            '_split_0.6_0.2_' + str(split) + '.npz'

        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']

        idx_train = np.where(train_mask == 1)[0]
        idx_val = np.where(val_mask == 1)[0]
        idx_test = np.where(test_mask == 1)[0]

    # if model =='link' or model =='linkx' or model =='mlse' or model =='mlse' or model == 'grcp':
    #     adj = adj + sp.eye(adj.shape[0])
    # elif "acm" in model or "ash" or "mlr" in model:
    #     pass 
    # else:
    #     adj = normalize(adj + sp.eye(adj.shape[0]))
        
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    if normfeat ==0:
        # features = normalize(features) 
        pass 
    else:
        features = normalize(features) 

    features = torch.FloatTensor(np.array(features.todense())) 
    labels = torch.LongTensor(np.where(labels))[1]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    return adj, features, labels, idx_train, idx_val, idx_test

def normalize_tensor(mx, eqvar=None):
    """
    Row-normalize sparse matrix
    """
    rowsum = torch.sum(mx, 1)
    if eqvar:
        r_inv = torch.pow(rowsum, -1 / eqvar).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

    else:
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

def my_normalize(mx):
    # D^(-1/2) A D^(-1/2)
    rowsum = torch.sum(mx, 1)
    r_inv = torch.pow(rowsum, -0.5).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0
    mx = torch.mm(torch.mm(mx, torch.diag(r_inv)), torch.diag(r_inv))
    return mx

# if acm version, get adj low, adj high, adj identity. 
def train_prep(adj_low_unnormalized, features, labels,args):
    
    
    if (args.model == "acmgcnp" or args.model == "acmgcnpp") and (
        args.structure_info == 1
    ):
        pass
    else:
        features = normalize_tensor(features)

    nnodes = labels.shape[0]
    if args.structure_info:
        adj_low = normalize_tensor(torch.eye(nnodes) + adj_low_unnormalized.to_dense())
        adj_high = (torch.eye(nnodes) - adj_low).to(device).to_sparse()
        adj_low = adj_low.to(device)
        adj_low_unnormalized = adj_low_unnormalized.to(device)
    else:
        adj_low = normalize_tensor(torch.eye(nnodes) + adj_low_unnormalized.to_dense())
        adj_high = (torch.eye(nnodes) - adj_low).to(device).to_sparse()
        adj_low = adj_low.to(device)
        adj_low_unnormalized = None

    if (args.model == "acmsgc") and (args.hops > 1):
        A_EXP = adj_low.to_dense()
        for _ in range(args.hops - 1):
            A_EXP = torch.mm(A_EXP, adj_low.to_dense())
        adj_low = A_EXP.to_sparse()
        del A_EXP
        adj_low = adj_low.to(device).to_sparse()

    return (
        adj_high,
        adj_low,
        adj_low_unnormalized,
    )
    
    
def graph_structural_similarity(adj):
    
    # give a adjcency matrix whose format is torch sparse 
    # edge idx 
    # the number of nodes
    
    # firstly, we calculate the mul(adj,adj.T)
    t1 = time.time()
    similarity_matrix =  torch.spmm(adj,adj.to_dense().t()).to_sparse()
    t2 = time.time()
    
    print("time simiarity matrix multiplication",t2-t1) 
    
    edge_idx = similarity_matrix.coalesce().indices()
    row, col = edge_idx
    
    print("num edges is ", row.size()[0])
    
    t1 = time.time()
    # iterate on edge idx  
    for i in range(row.size()[0]):
        row_idx  = row[i] 
        col_idx  = col[i]
        row1 = adj[row_idx]
        row2 = adj[col_idx]
        neighbor_union_basis = torch.sparse.sum(row1+row2)
        similarity_matrix._values()[i] =  similarity_matrix._values()[i] / neighbor_union_basis
    
    t2 = time.time()
    print("time num to prob", t2-t1) 
    
    return similarity_matrix 


def graph_class_structural_separability(similarity_matrix, labels,train_idx):
    
    # transform s
    similarity_matrix = similarity_matrix.to_dense()
    nclasses = labels.max().item() + 1
    
    class_idx_list= []
    
    t1 = time.time()
    for i in range(nclasses):
        train_idx_set =  set(train_idx.cpu().tolist()) 
        class_i_idx_set = set(np.where(labels.cpu().numpy()==i)[0].tolist() )
        
        class_i_idx_train = torch.tensor(list(train_idx_set & class_i_idx_set))
        class_idx_list.append(class_i_idx_train)
    t2 = time.time()
    print("time train class idx ",t2-t1)
    
    class_structural_separability_matrix = torch.zeros([nclasses,nclasses])
    
    t1 = time.time()
    for i in range(nclasses):
        for j in range(nclasses):
           
            class_i_idx_train = class_idx_list[i]
            class_j_idx_train = class_idx_list[j] 
            
            if class_i_idx_train.size()[0] == 0 or class_j_idx_train.size()[0] ==0:
                continue
            class_structural_similarity_matrix =  similarity_matrix[class_i_idx_train][:,class_j_idx_train] 
            class_structural_separability = class_structural_similarity_matrix.mean()
            class_structural_separability_matrix[i][j] = class_structural_separability 
    t2 = time.time()
    print("time train class avg similarity",t2-t1)
    
    
    return class_structural_separability_matrix


def full_load_data(train_time,dataset_name, splits_file_path=None,):
    
    graph_adjacency_list_file_path = os.path.join('new_data', dataset_name, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_name,
                                                            'out1_node_feature_label.txt')

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if dataset_name=='film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])

    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                            label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                            label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    features = np.array(
        [features['features'] for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    
    labels = np.array(
        [label['label'] for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    
    g = adj

    with np.load((splits_file_path+dataset_name+'_split_0.6_0.2_'+str(train_time)+'.npz')) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']
        
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': train_mask, 'idx_val': val_mask, 'idx_test': test_mask}
    return data



def partition_graph( reweight_adj ):
        
    reweight_adj_dense = reweight_adj
    nnodes = reweight_adj_dense.size()[0] 
    
    reweight_adj_low = torch.where(reweight_adj_dense.to(torch.float64) < 0., 0., reweight_adj_dense.to(torch.float64))
    reweight_adj_low[range(nnodes),range(nnodes)] = 1. 
    
    reweight_adj_high = torch.where(reweight_adj_dense.to(torch.float64) >=0., 0., reweight_adj_dense.to(torch.float64))
    reweight_adj_high[range(nnodes),range(nnodes)] =  1.
    
    reweight_adj_low = reweight_adj_low.to_sparse()
    reweight_adj_high = reweight_adj_high.to_sparse() 
    
    return  reweight_adj_low, reweight_adj_high


def get_confidence(output):
    output = torch.softmax(output, dim=1)
    confidence, pred_label = torch.max(output, dim=1)
    return confidence, pred_label 

def regenerate_pseudo_label(output, labels, idx_train, unlabeled_index, threshold, device):
    're-generate pseudo labels every stage'
    
    idx_train_list = idx_train.tolist()
    unlabeled_index_list = unlabeled_index.tolist()
    confidence, pred_label = get_confidence(output)
    
    index = torch.where(confidence > threshold)[0]
    pseudo_index = []
    pseudo_labels = labels.clone().to(device)

    for i in index:
        if i not in idx_train_list:
            pseudo_labels[i] = pred_label[i]
            pseudo_index.append(i)

    pseudo_index = torch.tensor(pseudo_index).cuda()
    idx_train = torch.cat([idx_train, pseudo_index] )

    return idx_train, pseudo_labels

def normalize_graph( adj_raw):
    # normalize adj here     
    deg = torch.sum(adj_raw,dim=1).to(torch.float)
    D_isqrt = (deg).pow(-0.5)
    D_isqrt_dense =  D_isqrt
    # D_isqrt_dense[D_isqrt_dense == float('inf')] = 0.
    D_isqrt_dense = torch.diag_embed(D_isqrt_dense) 
    D_isqrt = D_isqrt_dense
    A_dense = adj_raw
    
    DAD = D_isqrt_dense * A_dense * D_isqrt_dense
    return DAD 



# the following function is to visulaize 
def find_wrong_nodes( output, labels,idx_test ):
    
    preds = output.max(1)[1]
    correct = preds.eq(labels)
    
    wrong_nodes = torch.nonzero(correct==0).squeeze().tolist()
    return idx_test[wrong_nodes]

def nodes_degree(adj_raw,test,wn):
    deg = torch.sum(adj_raw.to_dense(),dim=1).to(torch.float)
    
    dt = deg[test].tolist()
    dt.sort()
        
    dw = deg[wn].tolist()
    
    if not isinstance(dw,list):
        dw = [dw]
        
    print(dw)
    
    return


def nodes_homophily(idx, labels, adj_raw):
    
    nclass =  labels.max().item() + 1 
    labels_m = F.one_hot(labels,num_classes = nclass).to(torch.float32).cuda()
    homophily_matrix  = torch.spmm(adj_raw.to(torch.float32), labels_m)
    homophily_matrix_normal = normalize_tensor(homophily_matrix) 
    homophily_vector = homophily_matrix_normal * labels_m
    homophily_vector = homophily_vector.sum(dim=-1)
    
    print(homophily_vector[idx].tolist()) 
    import pdb
    pdb.set_trace()
    return 



def loss_funcs(loss_name):
    assert loss_name in ["kl", "sq", "ce"]

    def kl_loss(src, dst):
        q = dst
        p = src
        
        loss = (p*p.log() - p*q.log()).sum(-1, keepdim=False)
        return loss

    def sq_loss(src, dst):
        p = dst
        q = src
        criterion = nn.MSELoss(reduction="none")
        loss = criterion(q, p)
        return loss

    def ce_loss(src, dst):
        p = dst
        q = src + EPS
        loss = -(p*q.log()).sum(-1, keepdim=False)
        return loss

    if loss_name == "kl":
        loss_func = kl_loss
    if loss_name == "sq":
        loss_func = sq_loss
    if loss_name == "ce":
        loss_func = ce_loss
    return loss_func


def consistency_loss(   pred1, pred2,    consistency_type='ce',temp=0.6  ):
    
    
    pred1 = F.softmax(pred1.clone())
    pred2 = F.softmax(pred2.clone())
    
    if consistency_type == 'kl':
        loss_func = loss_funcs('kl')
    elif consistency_type == 'ce':
        loss_func = loss_funcs('ce')
    elif consistency_type == 'sq':
        loss_func = loss_funcs('sq')
    
    pred_center = ((pred1+pred2)/2.).detach()
    
    # citeseer best 
    # pred_center =  (torch.pow(pred_center, 1./0.6) / torch.sum(torch.pow(pred_center, 1./0.6), dim=1, keepdim=True))
    
    pred_center =  (torch.pow(pred_center, 1./temp) / torch.sum(torch.pow(pred_center, 1./temp), dim=1, keepdim=True))
    
    closs = loss_func(pred1,pred_center)+loss_func(pred2,pred_center) 
    
    closs = closs.mean()
    return closs 


def exchange_neighbors(feats, labels, train_idx, adj_raw ):

    feats_target = feats.clone()
    
    nclass = labels.max().item() + 1 
    
    deg = torch.sum(adj_raw.to_dense(),dim=1).to(torch.float)
    
    
    # ranomly exchange nodes in the same class 
    for i in range(nclass):
        temp_class_idx = torch.where(labels==i)[0] 
        # import pdb
        # pdb.set_trace()
        # randomly choose pair of nodes 
        temp_class_idx_list = temp_class_idx.tolist() 
        
        temp_class_idx_list = list(set(temp_class_idx_list) & (set(train_idx.tolist())))
        
        temp_class_idx_list_copy = copy.deepcopy(temp_class_idx_list)     
        random.seed(42)
        random.shuffle(temp_class_idx_list)
        
        deg1 = deg[temp_class_idx_list_copy]
        deg2 = deg[temp_class_idx_list]
        same_class_same_deg_nodes_idx = torch.where(deg1==deg2)[0] 
        # repeat = {}
        # for a in temp_class_idx_list:
        #     repeat[a] = 0
        # for a in range(len(temp_class_idx_list_copy)):
        #     if temp_class_idx_list_copy[a] in repeat:
        #         del temp_class_idx_list_copy[a]
        #         del temp_class_idx_list[a]
        temp_class_idx_tensor_copy = torch.tensor(temp_class_idx_list_copy)
        temp_class_idx_tensor  = torch.tensor(temp_class_idx_list)
        temp_class_idx_list_copy = (temp_class_idx_tensor_copy[same_class_same_deg_nodes_idx]).tolist()
        temp_class_idx_list = (temp_class_idx_tensor[same_class_same_deg_nodes_idx]).tolist()
        
        feats_target[temp_class_idx_list_copy] = feats[temp_class_idx_list]
        # feats_target[temp_class_idx_list] = feats[temp_class_idx_list_copy]
        
    # to generate two graph, and keep consistency of this two graph
    return feats_target


# exchange nodes with same degree? 

def exchange_on_similarity(feats, labels, train_idx, adj_raw ):
    """1. cosine similarity"""
    feats_target = feats.clone()
    
    nclass = labels.max().item() + 1 
    
    deg = torch.sum(adj_raw.to_dense(),dim=1).to(torch.float)
    
    
    # ranomly exchange nodes in the same class 
    for i in range(nclass):
        temp_class_idx = torch.where(labels==i)[0] 
        # import pdb
        # pdb.set_trace()
        # randomly choose pair of nodes 
        temp_class_idx_list = temp_class_idx.tolist() 
        
        temp_class_idx_list = list(set(temp_class_idx_list) & (set(train_idx.tolist())))
        
        temp_class_idx_list_copy = copy.deepcopy(temp_class_idx_list)     
        # for m in temp_class_idx_list:
        #     neighbor1= []
        neighbor2 = []
        
        random.seed(42)
        random.shuffle(temp_class_idx_list)
        
        deg1 = deg[temp_class_idx_list_copy]
        deg2 = deg[temp_class_idx_list]
        same_class_same_deg_nodes_idx = torch.where(deg1==deg2)[0] 
        
        temp_class_idx_tensor_copy = torch.tensor(temp_class_idx_list_copy)
        temp_class_idx_tensor  = torch.tensor(temp_class_idx_list)
        change1 = []
        change2 = []
        """ method 1:use cosine or cdist as similarity metric"""
        for m in range(len(temp_class_idx_tensor_copy)):
            neighbor1 = torch.nonzero(adj_raw.to_dense()[temp_class_idx_tensor_copy[m]])
            neighbor2 = torch.nonzero(adj_raw.to_dense()[temp_class_idx_tensor[m]])
            # import pdb
            # pdb.set_trace()
            matrix1 = feats[neighbor1]
            matrix2 = feats[neighbor2]
            matrix1 = matrix1.view(matrix1.shape[0],matrix1.shape[2])
            matrix2 = matrix2.view(matrix2.shape[0],matrix2.shape[2])
            from sklearn.metrics.pairwise import cosine_similarity
            # similarity = np.sum(cosine_similarity(matrix1, matrix2))/(matrix1.shape[0]*matrix2.shape[0])
            similarity = torch.sum(torch.cdist(matrix1, matrix2))/(matrix1.shape[0]*matrix2.shape[0])
            print(f"cosine similarity:{similarity}")
        """method 2:use neighbor train label as similarity metric"""
        for m in range(len(temp_class_idx_tensor_copy)):
            neighbor1 = torch.nonzero(adj_raw.to_dense()[temp_class_idx_tensor_copy[m]])
            neighbor2 = torch.nonzero(adj_raw.to_dense()[temp_class_idx_tensor[m]])
            neighbor1 = neighbor1 & train_idx
            neighbor2 = neighbor2 & train_idx
            label1 = labels[neighbor1]
            label2 = labels[neighbor2]
            same = 0
            for l in range(nclass):
                tmp1 = 0
                tmp2 = 0
                import pdb
                pdb.set_trace()
                for t1 in label1:
                    if t1 == l:
                        tmp1 = tmp1 + 1
                for t2 in label2:
                    if t2 == l:
                        tmp2 = tmp2 + 1
                same = same + min(tmp1, tmp2)
            simi = same /(len(label1) + len(label2))
            print(simi)
            if simi > 0.4:
                change1.append(temp_class_idx_tensor[m])
                change2.append(temp_class_idx_tensor_copy[m])
                
        """method 3:use a linear layer to produce a embedding, and use the embedding as similarity metric"""
        
        # temp_class_idx_list_copy = (temp_class_idx_tensor_copy[same_class_same_deg_nodes_idx]).tolist()
        # temp_class_idx_list = (temp_class_idx_tensor[same_class_same_deg_nodes_idx]).tolist()
        
        # feats_target[temp_class_idx_list_copy] = feats[temp_class_idx_list]
        # feats_target[temp_class_idx_list] = feats[temp_class_idx_list_copy]
        feats_target[change1] = feats[change2]
        feats_target[change2] = feats[change1]
    # to generate two graph, and keep consistency of this two graph
    return feats_target
    


class Logger(object):
    def __init__(self, logname, now):
        path = os.path.join('log-files', now.split('_')[0])

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path,logname + '.txt')
        print('saving log to ', path)

        self.terminal = sys.stdout
        self.file = None

        self.open(path)

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def close(self):
        self.file.close()


def Kmeans_and_paint(data, type, dataset, i, model, acc):
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    # generate sample data
    n,c = data.shape
    data = data.cpu().detach().numpy()
    labels = []
    for m in range(c):
        labels.append(f'label {m}')
    # import pdb
    # pdb.set_trace()
    # apply k-means clustering
    kmeans = KMeans(n_clusters=c, random_state=0).fit(data)

    # apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data)
    # visualize the results
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=10,c=kmeans.labels_, )
    plt.xlim(-5,10)
    plt.ylim(-5,10)
    plt.legend(labels)
    # plt.show()
    plt.savefig(f"res_pics/{dataset}_{model}_split_{i}.png")
    plt.close()
def paint_distribution(dataset,labels,adj):
    """paint the label distribution """
    x = len(labels)
    tmp = torch.zeros((x,x))
    for i in range(x):
        for j in range(x):
            if labels[j] == labels[i]:
                tmp[i,j] = 1
            else:
                tmp[i,j] = -1
    return tmp*adj.to_dense()
    # plt.imshow(tmp, cmap='gray')
    # plt.colorbar()
    # plt.title("heat map of the labels")
    # plt.savefig(f"label on {dataset}.png")
    # plt.close()
def paint_heat_map(dataset,reweight, epoch,types):
    """ paint the heat map of reweight matrix at different epoch"""
    x, y = reweight.shape
    reweight = reweight.to_dense()
    if epoch > 200:
        import pdb
        pdb.set_trace()
    reweight = torch.nn.functional.softmax(reweight, dim=1)
    reweight = reweight.cpu().detach().numpy()
    plt.imshow(reweight, cmap='gray')
    plt.colorbar()
    plt.title(f"{types} heatmap of dataset {dataset} at epoch {epoch}")
    plt.savefig(f"heat_map/{types}_{dataset}_{epoch}.png")
    plt.close()
def paint_loss_change(reweight, dataset, interval, epoch, types, real_labels, norm_adj):
    """paint the loss change with epochs, every line indicate one kind of label"""
    from PIL import Image
    label_num = torch.max(real_labels)+1
    avg_loss = []
    reweight = reweight.to_dense()
    reweight = torch.nn.functional.softmax(reweight, dim=1).detach().numpy()
    each_label_num = []
    x, y = reweight.shape
    for i in range(label_num):
        avg_loss.append(0)
        each_label_num.append(0)
    for i in range(x):
        label = real_labels[i]
        each_label_num[label] += 1
        for j in range(y):
            if real_labels[j] == label:
                each_label_num[label] += 1
                avg_loss[label] += (norm_adj[i,i]-reweight[i,j])**2
    for i in range(label_num):
        avg_loss[i] /= each_label_num[i]
    return avg_loss
    if epoch != 0:
        pre_pic = Image.open(f'loss_pics/{types}_{dataset}_{epoch-interval}.png')
        pre_pic = np.asarray(pre_pic)
        for i in range(label_num):
            plt.plot(epoch, avg_loss[i].detach().numpy(), c=colors[i])
        plt.savefig(f"loss_pics/{types}_{dataset}_{epoch}.png")
        plt.close()
    else:
        for i in range(label_num):
            plt.plot(epoch, avg_loss[i].detach().numpy(), c=colors[i])
        plt.title("the loss variance with epoch")
        
        plt.savefig(f"loss_pics/{types}_{dataset}_{epoch}.png")
        plt.close()

def paint_loss(list, interval, epoch, dataset):
    epochs = []
    f1 = []
    s1 = []
    f2 = []
    s2 = []
    idx = 0
    label_num = len(list[0])
    for i in range(0, epoch, interval):
        epochs.append(i)
        f1.append(list[idx])
        idx = idx + 1
        s1.append(list[idx])
        idx = idx + 1
        f2.append(list[idx])
        idx = idx + 1
        s2.append(list[idx])
        idx = idx + 1
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'gray'] 
    f1 = np.array(f1)
    f2 = np.array(f2)
    s1 = np.array(s1)
    s2 = np.array(s2)
    labels = []
    for i in range(label_num):
        labels.append(f'label {i}')
    for i in range(label_num):
        plt.plot(epochs, f1[:,i], c=colors[i])
    plt.legend(labels)
    plt.savefig(f"loss_pics/{dataset}_feature_0.png")
    plt.close()
    for i in range(label_num):
        plt.plot(epochs, s1[:,i], c=colors[i])
    plt.legend(labels)
    plt.savefig(f"loss_pics/{dataset}_structure_0.png")
    plt.close()
    for i in range(label_num):
        plt.plot(epochs, f2[:,i], c=colors[i])
    plt.legend(labels)
    plt.savefig(f"loss_pics/{dataset}_feature_1.png")
    plt.close()
    for i in range(label_num):
        plt.plot(epochs, s2[:,i], c=colors[i])
    plt.legend(labels)
    plt.savefig(f"loss_pics/{dataset}_structure_1.png")
    plt.close()
    # i = 0
    # labels.append(f'label {i}')
    # plt.plot(epochs, f1[:,i], c=colors[i])
    # plt.legend(labels)
    # plt.savefig(f"loss_pics/{dataset}_feature_0.png")
    # plt.close()
    # plt.plot(epochs, s1[:,i], c=colors[i])
    # plt.legend(labels)
    # plt.savefig(f"loss_pics/{dataset}_structure_0.png")
    # plt.close()
    # plt.plot(epochs, f2[:,i], c=colors[i])
    # plt.legend(labels)
    # plt.savefig(f"loss_pics/{dataset}_feature_1.png")
    # plt.close()
    # plt.plot(epochs, s2[:,i], c=colors[i])
    # plt.legend(labels)
    # plt.savefig(f"loss_pics/{dataset}_structure_1.png")
    # plt.close()
    
def paint_reweight_distribution(adj, dataset, epoch, reweight, types):
    reweight = reweight.to_dense()
    reweight = reweight*adj.to_dense()
    reweight = reweight[reweight!=0].view(-1)
    # import pdb
    # pdb.set_trace()
    reweight = reweight.detach().cpu().numpy()
    plt.hist(reweight, bins=50)
    plt.savefig(f"distribution_pic/reweight_distribution_{types}_{dataset}_{epoch}.png")
    plt.close()

def find_reweight_heterophily(reweight, idx_test, output, label, log, name, dataset):
    output = output[idx_test]
    label = label[idx_test]
    reweight = reweight.to_dense().cpu().detach().numpy()
    reweight = reweight[label.cpu()]
    reweight = reweight.T[label.cpu()]
    homo = []
    heter = []
    heter_positive = 0
    heter_negative = 0
    homo_positive = 0
    homo_negative = 0
    for i in range(reweight.shape[0]):
        for j in range(reweight.shape[0]):
            if label[i] == label[j]:
                homo.append(reweight[i,j])
                if reweight[i, j] >0:
                    homo_positive = homo_positive + 1
                else:
                    homo_negative = homo_negative + 1
            else:
                heter.append(reweight[i,j])
                if reweight[i, j] >0:
                    heter_positive = heter_positive + 1
                else:
                    heter_negative = heter_negative + 1
    zero_count = np.sum(reweight==0)
    log.write(f"dataset:{dataset} test size:{len(label)} {name} total zero values:{zero_count}, total heter_positive values:{heter_positive}\n")
    log.write(f"total heter_negative values:{heter_negative}, total homo_positive values:{homo_positive}, total homo_negative values:{homo_negative}\n")
    log.write(f"dataset:{dataset} {name} homo average is {np.mean(homo)}, heter average is {np.mean(heter)}\n")
    
def find_reweight_asymetric(reweight, idx_test, output, label, log, name, dataset):
    # reweight = reweight.to_dense().cpu().detach().numpy()
    asym = 0
    counter = 0
    r1 = (reweight + reweight.T) / 2
    r2 = (reweight - reweight.T) / 2
    r1_norm = np.linalg.norm(r1,1)
    r2_norm = np.linalg.norm(r2,1)
    # import pdb; pdb.set_trace()
    asym = (1-(r1_norm - r2_norm) / (r1_norm + r2_norm))/2
    # print(asym)
    return asym, reweight


def sparse_mx_add_identity(adj, lambd = 1):
    # A = A + lambd * I sparse version
    nnodes = adj.size()[0]
    src = adj._indices()[0]
    dst = adj._indices()[1]
    value = adj._values()
    # import pdb;pdb.set_trace()
    diag_pos = torch.where((src-dst)==0)
    value[diag_pos] += lambd
    adj_idx = torch.stack([src,dst])
    adj = torch.sparse.FloatTensor(adj_idx, value, size=(nnodes, nnodes)).to(device)
    return adj

def test_similarity_acc(labels, adj):
    # check the reweight matrix judge right accuracy
    if labels==None:
        return
    # import pdb; pdb.set_trace()
    if adj.is_sparse:
        src = adj._indices()[0]
        dst = adj._indices()[1]
        value = adj._values()
        pos_src = src[value>0]
        pos_dst = dst[value>0]
        neg_src = src[value<0]
        neg_dst = dst[value<0]
        pos_right = 0
        neg_right = 0
        # import pdb; pdb.set_trace()
        pos_len = len(pos_src)
        neg_len = len(neg_src)
        # import pdb; pdb.set_trace()
        pos_right = sum(labels[pos_src[i]] == labels[pos_dst[i]] for i in range(pos_len))
        neg_right = sum(labels[neg_src[i]] != labels[neg_dst[i]] for i in range(neg_len))

        if(len(pos_src) > 0):
            print(f"pos_right: {pos_right/len(pos_src)}")
        if(len(neg_src) > 0):
            print(f"neg_right: {neg_right/len(neg_src)}")
    else:
        pos_src = torch.nonzero(adj>0)
        neg_src = torch.nonzero(adj<0)
        pos_right = 0
        neg_right = 0
        # import pdb; pdb.set_trace()
        pos_len = len(pos_src)
        neg_len = len(neg_src)
        # import pdb; pdb.set_trace()
        pos_right = sum(labels[pos_src[i]] == labels[pos_src[i]] for i in range(pos_len))
        neg_right = sum(labels[neg_src[i]] != labels[neg_src[i]] for i in range(neg_len))

        if(len(pos_src) > 0):
            print(f"pos_right: {pos_right/len(pos_src)}")
        if(len(neg_src) > 0):
            print(f"neg_right: {neg_right/len(neg_src)}")

def paint_hist(type,reweight, dataset,layer):
    # import pdb; pdb.set_trace()
    reweight = reweight[reweight!=0]
    plt.title(f"{dataset} {type} asymetric distribution ")
    plt.hist(reweight, bins=10)
    plt.savefig(f"heter_pic/heter_distribution_{dataset}_{layer}.png")
    

def khops_adj(nnodes, adj, max_khops):
    
    khop_adj_list = []
    strict_khop_adj_list = []

    for k in range(max_khops):
        if k >0:
            khop_adj = adj   
            for _ in range(k):
                khop_adj = torch.spmm(khop_adj,adj.to_dense())     
            khop_adj_list.append(khop_adj)    
        else:
            khop_adj_list.append(adj.to_dense())    
             
    for k in range(max_khops):   
        if k ==0:   
            strict_khop_adj_list.append(adj)    
        else: 
            condition = (khop_adj_list[k] > 0)    
            for t in range(k) :
                condition = condition & (khop_adj_list[t] == 0)    
            khop_strict_adj = torch.where(condition,1.,0.)
            khop_strict_adj[range(nnodes),range(nnodes)] = 0.
            strict_khop_adj_list.append(khop_strict_adj.to_sparse())
    
    for k in range(max_khops):
        khop_adj_list[k] = khop_adj_list[k].cuda()
        strict_khop_adj_list[k] = strict_khop_adj_list[k].cuda()
        
    return strict_khop_adj_list 
