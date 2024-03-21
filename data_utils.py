import os
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from scipy import sparse as sp
from sklearn.metrics import roc_auc_score, f1_score
from torch_sparse import SparseTensor
import gdown
import torch.nn as nn
import argparse
import copy
from torch_geometric.utils import remove_self_loops

import matplotlib.pyplot as plt


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx



def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


def to_planetoid(dataset):
    """
        Takes in a NCDataset and returns the dataset in H2GCN Planetoid form, as follows:

        x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ty => the one-hot labels of the test instances as numpy.ndarray object;
        ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        split_idx => The ogb dictionary that contains the train, valid, test splits
    """
    split_idx = dataset.get_idx_split('random', 0.25)
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    graph, label = dataset[0]

    label = torch.squeeze(label)

    print("generate x")
    x = graph['node_feat'][train_idx].numpy()
    x = sp.csr_matrix(x)

    tx = graph['node_feat'][test_idx].numpy()
    tx = sp.csr_matrix(tx)

    allx = graph['node_feat'].numpy()
    allx = sp.csr_matrix(allx)

    y = F.one_hot(label[train_idx]).numpy()
    ty = F.one_hot(label[test_idx]).numpy()
    ally = F.one_hot(label).numpy()

    edge_index = graph['edge_index'].T

    graph = defaultdict(list)

    for i in range(0, label.shape[0]):
        graph[i].append(i)

    for start_edge, end_edge in edge_index:
        graph[start_edge.item()].append(end_edge.item())

    return x, tx, allx, y, ty, ally, graph, split_idx


def to_sparse_tensor(edge_index, edge_feat, num_nodes):
    """ converts the edge_index into SparseTensor
    """
    num_edges = edge_index.size(1)

    (row, col), N, E = edge_index, num_nodes, num_edges
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]

    value = edge_feat[perm]
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(N, N), is_sorted=True)

    # Pre-process some important attributes.
    adj_t.storage.rowptr()
    adj_t.storage.csr2csc()

    return adj_t


def normalize(edge_index):
    """ normalizes the edge_index
    """
    adj_t = edge_index.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    return adj_t


def gen_normalized_adjs(dataset):
    """ returns the normalized adjacency matrix
    """
    row, col = dataset.graph['edge_index']
    N = dataset.graph['num_nodes']
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    D_isqrt = deg.pow(-0.5)
    D_isqrt[D_isqrt == float('inf')] = 0

    DAD = D_isqrt.view(-1,1) * adj * D_isqrt.view(1,-1)
    DA = D_isqrt.view(-1,1) * D_isqrt.view(-1,1) * adj
    AD = adj * D_isqrt.view(1,-1) * D_isqrt.view(1,-1)
    return DAD, DA, AD


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)


def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = y_pred.float()
        y_pred = F.softmax(y_pred, dim=-1)[:,1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                                
            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)

def update_T(output, idx_train, labels, T, device):
    output = torch.softmax(output, dim=1)
    T.requires_grad = True
    optimizer = torch.optim.Adam([T], lr=0.01, weight_decay=5e-4)
    mse_criterion = torch.nn.MSELoss().cuda()
    index = torch.where(idx_train)[0]
    nclass = labels.max().item() + 1
    for epoch in range(200):
        optimizer.zero_grad()
        loss = mse_criterion(output[index], T[labels[index]]) + mse_criterion(T, torch.eye(nclass).to(device))
        loss.backward()
        optimizer.step()
    T.requires_grad = False
    return T

@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, result=None, sampling=False, subgraph_loader=None):
    import time
    if result is not None:
        out = result
    else:

        model.eval()
        if not sampling:
            out = model(dataset)
        else:
            out = model.inference(dataset, subgraph_loader)
    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    return train_acc, valid_acc, test_acc, out

@torch.no_grad()
def pseudo_evaluate(model, dataset, train_idx, val_idx, test_idx, eval_func, result=None, sampling=False, subgraph_loader=None):
    import time 
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    if result is not None:
        out = result
    else:
        model.eval()
        a = time.time()
        if not sampling:
            out = model(dataset)
        else:
            out = model.inference(dataset, subgraph_loader)
        b = time.time()
        # print(b-a)

    train_acc = eval_func(
        dataset.label[train_idx], out[train_idx])
    valid_acc = eval_func(
        dataset.label[val_idx], out[val_idx])
    test_acc = eval_func(
        dataset.label[test_idx], out[test_idx])
    
    c = time.time()
     
    # print("total",c-a) 


    return train_acc, valid_acc, test_acc, out

def load_fixed_splits(dataset, sub_dataset):
    """ loads saved fixed splits for dataset
    """
    name = dataset
    if sub_dataset and sub_dataset != 'None':
        name += f'-{sub_dataset}'
    # import pdb; pdb.set_trace()
    if not os.path.exists(f'./data/splits/{name}-splits.npy'):
        try: assert dataset in splits_drive_url.keys()
        except:
            print(name)
        gdown.download(
            id=splits_drive_url[dataset], \
            output=f'./data/splits/{name}-splits.npy', quiet=False) 
    
    splits_lst = np.load(f'./data/splits/{name}-splits.npy', allow_pickle=True)
    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
    return splits_lst


dataset_drive_url = {
    'twitch-gamer_feat' : '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges' : '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
    'snap-patents' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia', 
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y', 
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ', 
    'wiki_views': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP', # Wiki 1.9M 
    'wiki_edges': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u', # Wiki 1.9M 
    'wiki_features': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK' # Wiki 1.9M
}

splits_drive_url = {
    'snap-patents' : '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N', 
    'pokec' : '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_', 
}

def uncertainty_dropout(dataset, adj, features, nclass, model, args, device):
    f_pass = 100
    out_list = []
    with torch.no_grad():
        for _ in range(f_pass):
            output = model(dataset)
            output = torch.softmax(output, dim=1)
            out_list.append(output)
        out_list = torch.stack(out_list)
        out_mean = torch.mean(out_list, dim=0)
        entropy = torch.sum(torch.mean(out_list * torch.log(out_list), dim=0), dim=1)
        Eentropy = torch.sum(out_mean * torch.log(out_mean), dim=1)
        bald = entropy - Eentropy
    return bald

def uncertainty_dropout_feature(adj, features, nclass, model, args, device):
    f_pass = 100
    out_list = []
    with torch.no_grad():
        model.eval()
        for _ in range(f_pass):
            features_tmp = features.clone()
            features_tmp = F.dropout(features_tmp, p = args.droprate)
            output = model(features_tmp, adj)
            output = torch.softmax(output, dim=1)
            out_list.append(output)
        out_list = torch.stack(out_list)
        out_mean = torch.mean(out_list, dim=0)
        entropy = torch.sum(torch.mean(out_list * torch.log(out_list), dim=0), dim=1)
        Eentropy = torch.sum(out_mean * torch.log(out_mean), dim=1)
        bald = entropy - Eentropy
    return bald

def regenerate_pseudo_label(output, labels, idx_train, train_idx_ag, unlabeled_index, threshold, device, nclass, sign=False):
    're-generate pseudo labels every stage'
    unlabeled_index = torch.where(unlabeled_index == True)[0]
    confidence, pred_label = get_confidence(output,nclass, sign)
    index = torch.where(confidence > threshold)[0]
    pseudo_index = []
    pseudo_labels, idx_train_ag = labels.clone().to(device), idx_train.clone().to(device)
    
    for i in index:
        if i not in train_idx_ag:
            pseudo_labels[i] = pred_label[i]
            pseudo_index.append(i)
            idx_train_ag[i] = True

        
    idx_pseudo = torch.zeros_like(idx_train)
    pseudo_index = torch.tensor(pseudo_index)

    if pseudo_index.size()[0] != 0:
        idx_pseudo[pseudo_index] = 1
    return idx_train_ag, pseudo_labels, idx_pseudo


def get_confidence(output, nclass,  with_softmax=False, device='cuda:0'):
    if nclass > 2:
        if not with_softmax:
            output = torch.softmax(output, dim=1)  
        confidence, pred_label = torch.max(output, dim=1)
        
    else:
        m = nn.Sigmoid().to(device)
        sigmoid_res = m(output)
        confidence, pred_label = torch.max(sigmoid_res, dim=1)     

    return confidence, pred_label

def weighted_cross_entropy(output, labels, bald, beta, nclass, sign=True):
    bald += 1e-6
    if sign:
        output = torch.softmax(output, dim=1)
    bald = bald / (torch.mean(bald) * beta)
    labels = F.one_hot(labels, nclass)
    loss = -torch.log(torch.sum(output * labels, dim=1))
    loss = torch.sum(loss * bald)
    loss /= labels.size()[0]
    return loss

def cross_entropy(output, labels, nclass, sign=True, device='cuda:0'):
    if nclass > 2:
        if sign:
            output = torch.softmax(output, dim=1)
        labels = F.one_hot(labels, nclass)
        loss = -torch.log(torch.sum(output * labels, dim=1))
        loss = torch.sum(loss)
        loss /= labels.size()[0]
    
    else:
        nclass = int(nclass)
        labels = labels
        m = nn.Sigmoid().to(device)
        sigmoid_res = m(output)
        loss = nn.BCELoss().to(device)
        loss = loss(sigmoid_res, labels)

    return loss

# def static_flip_graph(graph, pred,thres):
#     if pred != None:
#         edges = graph.edges()
        
#         src = edges[0].long()
#         dst = edges[1].long()
        
#         src_pred = pred[src]
#         dst_pred = pred[dst]
        
#         s = F.cosine_similarity(src_pred, dst_pred, dim=-1)
        
#         values, indexs = torch.sort(s)
        
#         flip = indexs[:int(thres*len(indexs))].cpu().tolist()
        
#         edge_weight = torch.ones(graph.edges()[0].size()[-1]).to(device)
#         edge_weight[flip] = -1. 
        
#         self_loop = torch.where(src==dst)[0]
#         edge_weight[self_loop] = 1.
        
#         graph.edata['w'] = edge_weight 
        
#         return graph 
        
#     else:
#         return graph
def static_flip_graph(graph, pred,thres, device, dgl_graph=None, weight = 10):
    if pred != None:
        edges = graph['edge_index']
        src = edges[0].long()
        dst = edges[1].long()
        
        src_pred = pred[src]
        dst_pred = pred[dst]
        
        s = F.cosine_similarity(src_pred, dst_pred, dim=-1)
        
        values, indexs = torch.sort(s)
        flip = indexs[:int(thres*len(indexs))].cpu().tolist()

        edge_weight = torch.ones(edges[0].size()[-1], dtype=torch.float).to(device)
        edge_weight[flip] = -1. 
        
        self_loop = torch.where(src==dst)[0]
        edge_weight[self_loop] = 1.
        
        graph['edge_weight'] = edge_weight 
        
        # nodes = torch.tensor([i for i in range(num_node)])
        # if dgl_graph is not None:
        #     degrees = dgl_graph.out_degrees()
        #     degrees = torch.tensor(degrees)
        #     degree = torch.cat((nodes.unsqueeze(1), degrees.unsqueeze(1)), dim=1)
        #     deg_with_index = sorted(degree, key=lambda x:x[1])
        #     index = [index for index, element in deg_with_index]
        #     value = [element for index, element in deg_with_index]
        #     # value_small = value[int(len(value)*0.2)]
        #     # value_large = value[int(len(value)*0.8)]
        #     # add_num = value_large - value_small
        #     index_small = index[:int(0.5*len(index))]
        #     # change_set = set(index_small) | set(flip)
        #     # change_set = list(change_set)
        #     change_set = index_small
        #     edge_weight[change_set] = edge_weight[change_set]*weight
        #     graph['edge_weight'] = edge_weight
        return graph 
        
    else:
        return graph

def random_flip_graph(dataset, flip_prob, device='cuda:0'):
    graph = dataset.graph
    edges = graph['edge_index']
    
    edge_weight = torch.ones(edges[0].size()[-1], dtype=torch.float).to(device)
    p1 = torch.rand_like(edge_weight).to(device)
    
    new_edge_weight = torch.where(p1 < flip_prob, -1.*edge_weight, edge_weight).to(device)
    
    src = edges[0].long()
    dst = edges[1].long()
    
    self_loop = torch.where(src==dst)[0]
    new_edge_weight[self_loop] = 1.
    
    graph['edge_weight'] = new_edge_weight 
    
    return graph

def update_flip_prob(flip_prob, best_output, graph, beta):
    
    edges = graph['edge_index']
    
    src = edges[0].long()
    dst = edges[1].long()

    
    src_pred = best_output[src]
    dst_pred = best_output[dst]
    s = F.cosine_similarity(src_pred, dst_pred, dim=-1)
    
    s = 2.*s -1. 
    flip_prob = flip_prob - beta*s
    
    flip_prob = torch.clamp(flip_prob, 0., 1.).to(device)
    
    return flip_prob

    

def confidence_vs_acc_homophily(name, adj, pred, labels,nclass,val_test_idx):
    
    mask_list = [ ] 
    acc_list =  [ ]
    homo_list = [ ]
    
    confidence, pred_label = get_confidence(pred,nclass, with_softmax=True)
    
    row = adj[0]
    col = adj[1] 
    
    edge_idx = torch.tensor(torch.vstack((row, col)), dtype=torch.long).contiguous()
    num_nodes = labels.shape[0]
    edge_index = remove_self_loops(edge_idx)[0]
    hs = torch.zeros(num_nodes)
    degs = torch.bincount(edge_index[0,:]).float()
    matches = (labels[edge_index[0,:]] == labels[edge_index[1,:]]).float()
    hs = hs.to(device)
    hs = hs.scatter_add(0, edge_index[0,:], matches.view(-1)) / (1e-10+degs)
    node_homo_vec =  hs
    
    pred_label = pred_label[val_test_idx] 
    labels = labels[val_test_idx]
    lable_sub = pred_label - labels.view(-1)
    
    confidence= confidence[val_test_idx]
    node_homo_vec= node_homo_vec[val_test_idx]
    
    for i in range(20):
        index1 = np.array(torch.where( confidence >0.05*(i) )[0].cpu())
        index2 = np.array(torch.where(  confidence <0.05*(i+1))[0] .cpu())
        index = torch.tensor(np.intersect1d(index1, index2)).cuda()
        
        if index.size()[0] !=0:
            if nclass > 2 or name not in ['yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'] :
                acc_b = torch.where(lable_sub[index] == 0, 1. , 0.)
                acc = acc_b.sum()/acc_b.size()[0] 
            else:
                acc = eval_rocauc(labels[index] ,pred[index]) 
            mask_list.append(index)
            homo_list.append( float(node_homo_vec[index].mean()) )  
            acc_list.append(float(acc))
        else:
            mask_list.append(index)
            
            homo_list.append( 0.  )  
            
            acc_list.append( 0.  )
    
    homo_arr = np.array(homo_list)
    acc_arr = np.array(acc_list)
    
    return homo_arr, acc_arr 
        
def draw_conf_acc_homo(name, conf, homo, acc, args):
    
    fig, ax = plt.subplots()
    ax.scatter(conf,homo,color='blue', label='node homophily' )
    ax.set_ylabel('Homophily',font={'family':'Arial', 'size':18})
    ax.set_xlabel('Confidence',font={'family':'Arial', 'size':18})
    ax.spines['right'].set_visible(False) 
    ax.set_ylim(0., 1.)

    z_ax = ax.twinx()
    z_ax.scatter(conf, acc,color='green',label='acc')
    
    z_ax.set_ylabel('Acc',font={'family':'Arial', 'size':18})
    z_ax.set_ylim(0., 1.)

    ax.grid(alpha=0.3,color='grey')
    ax.legend(loc='upper left',  fontsize=12)
    z_ax.legend(fontsize=12)
    
    plt.savefig('figures/test/{} {} confidence vs homophily & acc.png'.format(name,args.train_prop))
    
    plt.close()


# edges between two classes
#  return a nxn adjacency matrix
def edges_between_classes(args, adjl_raw, label, train_idx,pred):
    num_classes = int(label.max()+1)
    
    class_adjacency = torch.zeros([num_classes, num_classes]) 
    
    src,dst = adjl_raw[0], adjl_raw[1]
    
    confidence, pred_label = get_confidence(pred,num_classes)
    pred_label = pred_label.view(-1)
    pred_label = pred_label.long()
    

    edges_num = src.size()[0]
    
    for i in range(edges_num):
        class_adjacency[ pred_label[src[i]], pred_label[dst[i]]] += 1

    class_adjacency = class_adjacency.numpy()
    
    row_sum = class_adjacency.sum(axis=1)
    
    class_adjacency = class_adjacency/row_sum[:,np.newaxis]
    
    plt.title("{} pred edges between class nodes".format(args.dataset))

    for i in range(num_classes):
        for j in range(num_classes):
            text = plt.text(j, i, round(class_adjacency[i, j],2), ha="center", va="center", color="w")

    plt.imshow(class_adjacency)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('figures/'+ '{}_pred_edges_between_class_nodes.png'.format(args.dataset))
    
    plt.close()
    
    # calculate the 
    
    num_classes = int(label.max()+1)
    
    class_adjacency = torch.zeros([num_classes, num_classes]) 
    
    src,dst = adjl_raw[0], adjl_raw[1]
    
    edges_num = src.size()[0]
    
    for i in range(edges_num):
        class_adjacency[ label[src[i]], label[dst[i]]] += 1

    class_adjacency = class_adjacency.numpy()
    
    row_sum = class_adjacency.sum(axis=1)
    
    class_adjacency = class_adjacency/(1e-10+row_sum[:,np.newaxis])
    
    plt.title("{} edges between class nodes".format(args.dataset))

    for i in range(num_classes):
        for j in range(num_classes):
            text = plt.text(j, i, round(class_adjacency[i, j],2), ha="center", va="center", color="w")

    plt.imshow(class_adjacency)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('figures/'+ '{}_edges_between_class_nodes.png'.format(args.dataset))
    
    plt.close()
    
    class_train_adjacency = torch.zeros([num_classes, num_classes]) 
    
    train_edge_list = []
    for i in range(edges_num):
        if src[i] in train_idx and dst[i] in train_idx:
            train_edge_list.append(i)
    
    src_train = src[train_edge_list]
    dst_train = dst[train_edge_list]
    
    edges_num = src_train.size()[0]
    
    for i in range(edges_num):
        class_train_adjacency[ label[src_train[i]], label[dst_train[i]]] += 1

    class_train_adjacency = class_train_adjacency.numpy()
    
    row_sum = class_train_adjacency.sum(axis=1)
    
    class_train_adjacency = class_train_adjacency/(1e-10+row_sum[:,np.newaxis])
    
    
    plt.title("{} train edges {} between class nodes".format(args.dataset, edges_num))

    for i in range(num_classes):
        for j in range(num_classes):
            text = plt.text(j, i, round(class_train_adjacency[i, j],2), ha="center", va="center", color="w")

    plt.imshow(class_train_adjacency)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('figures/'+ '{}_{}_train_edges_between_class_nodes.png'.format(args.dataset, edges_num))
    
    plt.close()
    
def confidence_distribution(args, pred, labels, val_test_idx , nclass):
    
    confidence, pred_label = get_confidence(pred,nclass, True)
    
    pred_label = pred_label.view(-1)[val_test_idx]
    labels = labels.view(-1)[val_test_idx]
    
    correct_idx = torch.where(pred_label==labels)[0]
    incorrect_idx = torch.where(pred_label!=labels)[0]
    
    conf_cor = confidence[correct_idx].cpu().numpy()
    conf_icor = confidence[incorrect_idx].cpu().numpy()
    
    fig, ax = plt.subplots()
    
    x_multi = [conf_cor, conf_icor]
    
    n_bins = 20
    colors = ['blue','red']
    labels = ['correct','incorrect']
    
    ax.hist(x_multi, n_bins, histtype='bar', color=colors, label=labels)
    ax.legend()
    ax.set_xlabel('node confidence')
    ax.set_ylabel('nodes number')
    ax.set_title('node confidence distribution')
    
    plt.savefig('figures/test/'+ '{} conf_distribution_{}.png'.format(args.dataset, args.train_prop))
    
    plt.close()
    

def regenerate_pseudo_label_check(output, labels, idx_train, unlabeled_index, threshold, device, nclass, sign=False):
    're-generate pseudo labels every stage'
    unlabeled_index = torch.where(unlabeled_index == True)[0]
    
    confidence, pred_label = get_confidence(output,nclass, sign)
    index = torch.where(confidence > threshold)[0]
    
    pseudo_index = []
    pseudo_labels, idx_train_ag = labels.clone().to(device), idx_train.clone().to(device)
    
    
    # randomly choice from the 
    
    for i in index:
        if i not in idx_train:
            pseudo_labels[i] = pred_label[i]
            if i in unlabeled_index:
                idx_train_ag[i] = True
                pseudo_index.append(i)
    idx_pseudo = torch.zeros_like(idx_train)
    pseudo_index = torch.tensor(pseudo_index)
    if pseudo_index.size()[0] != 0:
        idx_pseudo[pseudo_index] = 1
    return idx_train_ag, pseudo_labels, idx_pseudo 


def get_class_train_adj(args, labels, nclass, edges):
    src=  edges[0] 
    dst = edges[1] 
    
    class_train_adjacency = torch.zeros([nclass, nclass]) 
    
    labels = labels.long()
    src= src.long()
    dst=dst.long()
    
    if nclass==1:
        nclass=2
    
    for i in range(nclass):
        for j in range(nclass):
            index = np.array(torch.where(labels[src]==i )[0].cpu())
            index2 = np.array(torch.where(  labels[dst] == j )[0] .cpu())
            index = torch.tensor(np.intersect1d(index, index2))
            class_train_adjacency[i,j] = float(index.size()[0])
    
    # normalize the class pred adjacency by row
    class_train_adjacency = class_train_adjacency.numpy()
    row_sum = class_train_adjacency.sum(axis=1)
    class_train_adjacency = class_train_adjacency/(1e-10+row_sum[:,np.newaxis])
    
    return class_train_adjacency 


def transition_loss(args, pred, nclass, edges, class_train_adjacency):
    
    class_pred_adjacency = (pred.softmax(dim=-1).transpose(0,1) @ pred.softmax(dim=-1))
    
    row_sum = class_pred_adjacency.sum(dim=1)
    class_pred_adjacency = class_pred_adjacency/(1e-10+row_sum) 
    
    # decrease the diagnal element of class pred adjacency
    diag_avg =  torch.diag(class_pred_adjacency).sum() /nclass 
    others_avg = (class_pred_adjacency.sum() - torch.diag(class_pred_adjacency).sum()) /(nclass*(nclass-1))
    dediag_loss = torch.abs(diag_avg-others_avg)
    
    class_train_adjacency = torch.tensor(class_train_adjacency).to(device)
    tt_loss =  torch.abs((class_pred_adjacency- class_train_adjacency).mean())
    
    loss = args.tt_coe* tt_loss + args.dediag_coe* dediag_loss

    return loss 


def acc_vs_factors(args, pred, nclass, edges, labels, index, device, ob_type='node'):
    
    name = args.dataset 
    train_idx = index[0]
    val_idx   = index[1]
    test_idx  = index[2] 
    
    node_confidence, pred_label = get_confidence(pred,nclass, with_softmax=True) 
    node_degree = torch.bincount(edge_index[0,:]).float()
    
    matches = (pred_label[edge_index[0,:]] == pred_label[edge_index[1,:]]).float()
    node_predicted_homophily = hs.scatter_add(0, edge_index[0,:], matches.view(-1)) / (1e-10+degs)
    
    val_test_idx = torch.cat([val_idx, test_idx]).to(device)
    
    labels = labels[val_test_idx]
    pred_label = pred_label[val_test_idx]
    node_predicted_homophily = node_predicted_homophily[val_test_idx] 
    node_confidence = node_confidence[val_test_idx] 
    node_degree = node_degree[val_test_idx]
    lable_sub = pred_label - labels.view(-1)
    
    mask_list = [] 
    homo_list = [] 
    degree_list = []
    
    acc_list = []
    
    for i in range(20):
        index1 = np.array(torch.where( node_confidence >0.05*(i) )[0].cpu())
        index2 = np.array(torch.where(  node_confidence <0.05*(i+1))[0] .cpu())
        index = torch.tensor(np.intersect1d(index1, index2)).cuda()
        
        if index.size()[0] !=0:
            if nclass > 2 or name not in ['yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'] :
                acc_b = torch.where(lable_sub[index] == 0, 1. , 0.)
                acc = acc_b.sum()/acc_b.size()[0] 
            else:
                acc = eval_rocauc(labels[index] ,pred[index]) 
                
            mask_list.append(index)
            
            homo_list.append( float(node_predicted_homophily[index].mean()) )  
            
            acc_list.append(float(acc))
        
        else:
            mask_list.append(index)
            
            homo_list.append(0.)  
            
            acc_list.append(0.)
    
    homo_arr = np.array(homo_list)
    acc_arr = np.array(acc_list)
    
    if ob_type == 'node':
        pass
    
    if ob_type=='node':
        pass

def class_acc(pred, labels):
    nclass = int(labels.max())+1 
    acc_list = []
    for i in range(nclass):
        idx = torch.where( labels==i )[0]
        class_acc = accuracy(pred[idx], labels[idx])
        acc_list.append(class_acc)
        
    return acc_list

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    
    acc = float(torch.where(preds==labels.view(-1), 1. , 0.).sum()/ preds.shape[0])
    
    return acc 

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, weight=None):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.weight = weight

#     def forward(self, inputs, targets):
#         ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets) 
#         pt = torch.exp(-ce_loss)  
#         focal_loss = (1 - pt) ** self.gamma * ce_loss 
#         return focal_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1  
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1) 
        preds_logsoft = torch.log(preds_softmax)
        
        #focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1)) 
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) 

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
    
def get_memory_usage():
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_memory = memory_info.used
    return used_memory / (1024*1024)

def re_split(graph, split, ratio_train=0.5, ratio_val=0.25):
    import dgl
    ori_train = copy.deepcopy(split['train'])
    ori_val = copy.deepcopy(split['valid'])
    ori_test = copy.deepcopy(split['test'])
    total_number = len(ori_train) + len(ori_val) + len(ori_test)
    train_number = int(total_number*ratio_train)
    val_number = int(total_number*ratio_val)
    # print(train_number, val_number)
    dgl_graph = dgl.graph((graph['edge_index'][0], graph['edge_index'][1]))
    degrees = dgl_graph.in_degrees()
    sorted_degrees, indices = torch.sort(degrees, descending=True)
    train_idx = indices[:train_number]
    val_idx = indices[train_number:train_number+val_number]
    test_idx = indices[train_number+val_number:]
    split['train'] = train_idx
    split['valid'] = val_idx
    split['test'] = test_idx
    return split

def deep_walk():
    pass

def random_sampling():
    pass

def random_projection():
    pass

def strap():
    pass

def paint_new_fig(number, acc,confidence,precision,args, std = 0):
    number_max = max(number)
    acc_max = max(acc)
    confi_max = max(confidence)
    precision_max = max(precision)
    number_ori = copy.deepcopy(number)
    acc_ori = copy.deepcopy(acc)
    confidence_ori = copy.deepcopy(confidence)
    precision_ori = copy.deepcopy(precision)
    number = [ele / number_max for ele in number]
    acc = [ele / acc_max for ele in acc]
    confidence = [ele / confi_max for ele in confidence]
    precision = [ele / precision_max for ele in precision]
    list2 = number
    list3 = acc
    list4 = precision
    list5 = confidence
    assert  len(list2) == len(list3) == len(list4) == len(list5)

    # 设置柱状图的位置和宽度
    bar_width = 0.20
    index = np.arange(len(list2))

    # 创建一个画布
    plt.figure(figsize=(10, 8))

    # 绘制柱状图
    
    bars = plt.bar(index + bar_width, list2, bar_width, label='label number')
    for idx, bar in enumerate(bars):
        yval = round(number_ori[idx],2 )
        height = round(bar.get_height(),2 )
        plt.text(bar.get_x() + bar.get_width()/2, height, yval, ha='center', va='bottom')
    bars = plt.bar(index + 2 * bar_width, list3, bar_width, label='label acc')
    for idx, bar in enumerate(bars):
        yval = round(acc_ori[idx],2 )
        height = round(bar.get_height(),2 )
        plt.text(bar.get_x() + bar.get_width()/2, height, yval, ha='center', va='bottom')
    bars = plt.bar(index + 3 * bar_width, list4, bar_width, label='label precision')
    for idx, bar in enumerate(bars):
        yval = round(precision_ori[idx],2 )
        height = round(bar.get_height(),2 )
        plt.text(bar.get_x() + bar.get_width()/2, height, yval, ha='center', va='bottom')
    bars = plt.bar(index + 4 * bar_width, list5, bar_width, label='label confidence')
    for idx, bar in enumerate(bars):
        yval = round(confidence_ori[idx].item(),2 )
        height = round(bar.get_height(),2 )
        plt.text(bar.get_x() + bar.get_width()/2, height, yval, ha='center', va='bottom')
    # 添加标题和坐标轴标签
    plt.xlabel('labels')
    plt.ylabel('Values')
    plt.title(f'{args.method}_{args.dataset}_{args.num_layers} label distribution, std:{std*100:.2f}')
    plt.xticks(index + bar_width, [f'Label {i+1}' for i in range(len(list2))])

    # 添加图例
    plt.legend(loc='lower left')

    # 显示图表
    plt.savefig(f"save_fig/{args.method}_{args.dataset}_adj_order_{args.adj_order}_num_layers_{args.num_layers}.png")
    plt.close()
# def paint_new_fig(number, acc,confidence,precision,args, std = 0):
def paint_new_fig_analysis(train_acc, val_acc,test_acc,confidence1,args, std = 0):
    # train, val, test, confidence
    number = train_acc
    acc = val_acc
    precision = test_acc
    confidence = confidence1
    # number_max = max(number)
    # acc_max = max(acc)
    # confi_max = max(confidence)
    # precision_max = max(precision)
    # number_ori = copy.deepcopy(number)
    # acc_ori = copy.deepcopy(acc)
    # confidence_ori = copy.deepcopy(confidence)
    # precision_ori = copy.deepcopy(precision)
    # number = [ele / number_max for ele in number]
    # acc = [ele / acc_max for ele in acc]
    # confidence = [ele / confi_max for ele in confidence]
    # precision = [ele / precision_max for ele in precision]
    list2 = number
    list3 = acc
    list4 = precision
    list5 = confidence
    assert  len(list2) == len(list3) == len(list4) == len(list5)
    # import pdb; pdb.set_trace()
    # 设置柱状图的位置和宽度
    bar_width = 0.20
    index = np.arange(len(list2))

    # 创建一个画布
    plt.figure(figsize=(10, 8))

    # 绘制柱状图
    
    bars = plt.bar(index + bar_width, list2, bar_width, label='train acc')
    for idx, bar in enumerate(bars):
        yval = round(number[idx],2 )
        height = round(bar.get_height(),2 )
        plt.text(bar.get_x() + bar.get_width()/2, height, yval, ha='center', va='bottom')
    bars = plt.bar(index + 2 * bar_width, list3, bar_width, label='val acc')
    for idx, bar in enumerate(bars):
        yval = round(acc[idx],2 )
        height = round(bar.get_height(),2 )
        plt.text(bar.get_x() + bar.get_width()/2, height, yval, ha='center', va='bottom')
    bars = plt.bar(index + 3 * bar_width, list4, bar_width, label='test acc')
    for idx, bar in enumerate(bars):
        yval = round(precision[idx],2 )
        height = round(bar.get_height(),2 )
        plt.text(bar.get_x() + bar.get_width()/2, height, yval, ha='center', va='bottom')
    bars = plt.bar(index + 4 * bar_width, list5, bar_width, label='label confidence')
    for idx, bar in enumerate(bars):
        yval = round(confidence[idx].item(),2 )
        height = round(bar.get_height(),2 )
        plt.text(bar.get_x() + bar.get_width()/2, height, yval, ha='center', va='bottom')
    # 添加标题和坐标轴标签
    plt.xlabel('labels')
    plt.ylabel('Values')
    # plt.title(f'{args.method}_{args.dataset}_{args.num_layers} label distribution, std:{std*100:.2f}')
    plt.xticks(index + bar_width, [f'Label {i+1}' for i in range(len(list2))])

    # 添加图例
    plt.legend(loc='lower left')

    # 显示图表
    plt.savefig(f"save_fig/{args.method}_{args.dataset}_adj_order_{args.adj_order}_num_layers_{args.num_layers}_analysis.png")
    plt.close()

def t_sne_analyis(args, model, dataset,features=None, out_size = 5000, use_two_labels = None, label_accuracy=None, split=None):
    import torch
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    # from tsnecuda import TSNE
    model.eval()
    num_label = max(dataset.label).item() + 1
    if use_two_labels == None:
        label_list = [i for i in range(num_label)]
    else:
        label_list = use_two_labels
    color_set = ['red', 'green', 'blue', 'yellow', 'brown']
    # final_out, my_out = model(dataset, find_each_layer=True)
    # # y_pred = final_out.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    # # if features == None:
    # #     features = features.detach().cpu().numpy()
    # # else:
    # #     features = my_out[0].to_dense().detach().cpu().numpy()
    # # labels = dataset.label
    # tsne = TSNE(n_components=2, perplexity=50)

    # # if features.shape[0] > out_size:
    # #     indices = torch.randperm(features.shape[0])[:out_size]
    # #     features = features[indices]
    # #     labels = labels[indices]
    # # features = F.softmax(torch.tensor(features), dim=-1).detach().cpu().numpy()
    # # features = tsne.fit_transform(features)
    # # for label in label_list:
    # #     label_idx = torch.where(labels==label)[0]
    # #     label_i = features[label_idx.detach().cpu()]
    # #     plt.scatter(label_i[:, 0], label_i[:, 1],marker="x",s=2.5, c=color_set[label],alpha=0.5, label=f"label {label}")
        
    # # plt.title('feature output')
    # # plt.legend(loc='best')
    # # if len(label_list)==2:
    # #     plt.savefig(f"new_tsne_analysis/{args.dataset}_{args.method}_feature_adj_order_{args.adj_order}_{label_list[0]}_{label_list[1]}.png")
    # # else:
    # #     plt.savefig(f"new_tsne_analysis/{args.dataset}_{args.method}_adj_order_{args.adj_order}_feature.png")
    # # plt.close()
    # # tsne = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(X)
    # idx = 0
    # for out in my_out:
    #     if isinstance(out, list):
    #         for idx1, out1 in enumerate(out):
    #             labels = dataset.label
    #             out1 = out1.detach().cpu().numpy()
    #             if out1.shape[0] > out_size:
    #                 indices = torch.randperm(out1.shape[0])[:out_size]
    #                 out1 = out1[indices]
    #                 labels = labels[indices]
    #             try:
    #                 data_2d = tsne.fit_transform(out1)
    #             except:
    #                 continue

    #             for label in label_list:
    #                 label_idx1 = torch.where(labels==label)[0]
    #                 label_i = data_2d[label_idx1.detach().cpu()]
    #                 plt.scatter(label_i[:, 0], label_i[:, 1],marker="x",s=2.5, c=color_set[label],alpha=0.5, label=f"label {label}")
                 
    #             plt.legend(loc='best')
    #             plt.title(f'{args.dataset} {args.method} hop {idx} layer {idx1} output')
    #             if len(label_list)==2:
    #                 plt.savefig(f"new_tsne_analysis/{args.dataset}_{args.method}_adj_order_{args.adj_order}_hop_{idx}_layer_{idx1}_{label_list[0]}_{label_list[1]}.png")
    #             else:
    #                 plt.savefig(f"new_tsne_analysis/{args.dataset}_{args.method}_adj_order_{args.adj_order}_hop_{idx}_layer_{idx1}.png")
                
    #             plt.close()
    #     else:
    #         labels = dataset.label
    #         # out = F.softmax(torch.tensor(out.to_dense()), dim=-1)
    #         out = out.to_dense()
    #         out = out.detach().cpu().numpy()
    #         if out.shape[0] > out_size:
    #             indices = torch.randperm(out.shape[0])[:out_size]
    #             out = out[indices]
    #             labels = labels[indices]
    #         try:
    #             data_2d = tsne.fit_transform(out)
    #         except:
    #             continue

    #         for label in label_list:
    #             label_idx = torch.where(labels==label)[0]
    #             label_i = data_2d[label_idx.detach().cpu()]
    #             plt.scatter(label_i[:, 0], label_i[:, 1],marker="x",s=2.5, c=color_set[label],alpha=0.5, label=f"label {label}")
            
    #         plt.legend(loc='lower left')
    #         plt.title(f'{args.dataset} {args.method} layer {idx} output')
    #         plt.savefig(f"new_tsne_analysis/{args.dataset}_{args.method}_adj_order_{args.adj_order}_layer_{idx}.png")
            
    #         plt.close()
    #         idx += 1
    # import pdb; pdb.set_trace()
    final_out = model(dataset)[split['test']]
    labels = dataset.label[split['test']]
    # final_out = F.softmax(torch.tensor(final_out))
    final_out = final_out.detach().cpu().numpy()
    tsne = TSNE()
    if final_out.shape[0] > out_size:

        indices = torch.randperm(final_out.shape[0])[:out_size]
        final_out = final_out[indices]
        labels = labels[indices]
    final_out = tsne.fit_transform(final_out)
    for label in label_list:
        label_idx = torch.where(labels==label)[0]
        label_i = final_out[label_idx.detach().cpu()]
        plt.scatter(label_i[:, 0], label_i[:, 1],marker="x",s=5, c=color_set[label],alpha=1, label=f"label {label}")
    
    # plt.title('final_layer output')
    plt.legend(loc='best')
    if len(label_list)==2:
        plt.savefig(f"new_tsne_analysis/{args.dataset}_{args.method}_adj_order_{args.adj_order}_label_{label_list[0]}_label_{label_list[1]}_final.png")
    else:
        plt.savefig(f"new_tsne_analysis/{args.dataset}_{args.method}_adj_order_{args.adj_order}_final.png")
    plt.close()

def paint_hist(input):
    # import pdb; pdb.set_trace()
    input = input[torch.where(input<5)]
    input = input.detach().cpu().numpy()
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    data_sorted = np.sort(input)  # 首先对数据进行排序
    interval = len(data_sorted) // 20  # 计算组距
    x = [data_sorted[i*interval] for i in range(20)]
    x.append(data_sorted[-1])
    cumulative = np.cumsum(np.ones_like(data_sorted)) / len(data_sorted)  # 计算累计频率
    y = [cumulative[i*interval] for i in range(20)]
    y.append(cumulative[-1])
    print(f"weight distribution: {x}")
    print(f"cumulative distribution: {y}")
    # import pdb; pdb.set_trace()
    # 加载并显示图片
    # img = mpimg.imread('distribution.png')
    # plt.imshow(img)
    # plt.xlim(0, 2)
    plt.plot(data_sorted, cumulative, marker='.', linestyle='-', color='b')
    # fig, ax = plt.subplots()
    # ax.hist(input, bins=10, alpha=0.5, color='blue', edgecolor='red', cumulative=True, density=True,histtype='step',label='snap-patents')
    plt.xlabel('weight')
    plt.ylabel('number')
    plt.legend(loc='best')
    plt.savefig("distribution.png")
    plt.close()
    
def count_label_homo(dataset, order=1,dataname='squirrel'):
    import dgl
    # import pdb; pdb.set_trace()
    c = 5
    edge_index = dataset.graph['edge_index']
    num_nodes = dataset[0][0]['num_nodes']
    for i in range(order):
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            row = row-row.min() # for sampling
            A = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes)).to_torch_sparse_coo_tensor()
        elif isinstance(edge_index, SparseTensor):
            A = edge_index.to_torch_sparse_coo_tensor()
        A_o = A
        for k in range(i):
            A = A @ A_o
        dgl_graph = dgl.graph((A._indices()[0], A._indices()[1]))
        # total_homo = dgl.node_homophily(graph, dataset.label.squeeze(1))
        dgl_graph = dgl.remove_self_loop(dgl_graph)
        homo_list = []
        label_homo_list = []
        for node in range(dgl_graph.number_of_nodes()):
            neighbors = dgl_graph.successors(node)
            
            if len(neighbors) == 0:
                homo_list.append(0)
                continue
            same_category_count = torch.sum(dataset.label[neighbors] == dataset.label[node])
            homophily_ratio = same_category_count.item() / len(neighbors)
            homo_list.append(homophily_ratio)
        for l in range(c):     
            # tmp = dataset.label[split_idx['test']]
            label_cur = torch.where(dataset.label==l)[0]
            label_homo = torch.mean(torch.tensor(homo_list)[label_cur]).item()
            label_homo_list.append(label_homo)
        print(f"{dataname} order {i+1} label_homo_list is {label_homo_list}")
        print(f"avg homo is {np.mean(label_homo_list)}")

