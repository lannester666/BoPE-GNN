import argparse
import sys
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_geometric.data import NeighborSampler
from torch_scatter import scatter
import copy
from logger import Logger, SimpleLogger
from dataset import load_nc_dataset
from correct_smooth import double_correlation_autoscale, double_correlation_fixed
from data_utils import *
from parse import parse_method, parser_add_main_args 
import faulthandler; faulthandler.enable()
import pickle
from dgl.nn import DeepWalk
import time
from torch.optim import SparseAdam
import dgl
from torch.utils.data import DataLoader
def main():
    torch.manual_seed(3407)
    np.random.seed(0)

    ### Parse args ###
    parser = argparse.ArgumentParser(description='General Training Pipeline')
    parser_add_main_args(parser)
    args = parser.parse_args()
    print(args)
    device = args.device

    ### Load and preprocess data ###
    dataset = load_nc_dataset(args.dataset, args.sub_dataset)

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(device)

    if args.rand_split or args.dataset in ['ogbn-proteins', 'wiki']:
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                    for _ in range(args.runs)]
    else:
        split_idx_lst = load_fixed_splits(args.dataset, args.sub_dataset)

    if args.dataset == 'ogbn-proteins':
        if args.method == 'mlp' or args.method == 'cs':
            dataset.graph['node_feat'] = scatter(dataset.graph['edge_feat'], dataset.graph['edge_index'][0],
                dim=0, dim_size=dataset.graph['num_nodes'], reduce='mean')
        else:
            dataset.graph['edge_index'] = to_sparse_tensor(dataset.graph['edge_index'],
                dataset.graph['edge_feat'], dataset.graph['num_nodes'])
            dataset.graph['node_feat'] = dataset.graph['edge_index'].mean(dim=1)
            dataset.graph['edge_index'].set_value_(None)
        dataset.graph['edge_feat'] = None

    n = dataset.graph['num_nodes']
    # infer the number of classes for non one-hot and one-hot labels
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]

    # whether or not to symmetrize matters a lot!! pay attention to this
    # e.g. directed edges are temporally useful in arxiv-year,
    # so we usually do not symmetrize, but for label prop symmetrizing helps
    if not args.directed and args.dataset != 'ogbn-proteins':
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

    dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        dataset.graph['edge_index'].to(
            device), dataset.graph['node_feat'].to(device)
    train_loader, subgraph_loader = None, None

    print(f"num nodes {n} | num classes {c} | num node feats {d}")

    # using rocauc as the eval function
    if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
        criterion = nn.BCEWithLogitsLoss()
        eval_func = eval_rocauc
    else:
        criterion = nn.NLLLoss()
        eval_func = eval_acc

    logger = Logger(args.runs, args)
    
    if args.transition_loss != 'none':
        class_train_adjacency = get_class_train_adj(args, dataset.label.squeeze(1), c, dataset.graph['edge_index'])
    
    ori_graph = copy.deepcopy(dataset.graph)
    ori_graph['edge_weight'] = torch.ones_like(dataset.graph['edge_index'][-1], dtype=torch.float)
    features0 = dataset.graph['node_feat'].clone()
    num_nodes = dataset.graph['num_nodes']
    edge_index = dataset.graph['edge_index']

    if isinstance(edge_index, torch.Tensor):
        row, col = edge_index
        row = row-row.min() # for sampling
        A1 = SparseTensor(value = torch.tensor([1. for i in range(row.shape[0])]).to(edge_index.device), row=row, col=col, sparse_sizes=(num_nodes, num_nodes)).to_torch_sparse_coo_tensor().detach().cpu()
    ### Training loop ###
    ##### ProNE #####
    A_o = A1
    t1 = time.time()
    for i in range(args.adj_order - 1):
        A1 = A1 @ A_o
    dgl_graph = dgl.graph((A1._indices()[0], A1._indices()[1]))
    from models import ProNE
    from models import save_embedding
    args.step = 10
    args.mu = 0.2
    args.theta = 0.5
    args.emb1 = f'save_embd/{args.dataset}_prone_embd_{args.tebd_dim}.pt'
    args.emb2 = f'save_embd/{args.dataset}_prone_embd_{args.tebd_dim}_enhanced.pt'
    P_model = ProNE(dgl_graph,args.emb1, args.emb2, args.tebd_dim)
    features_matrix = P_model.pre_factorization(P_model.matrix0, P_model.matrix0)
    embeddings_matrix = P_model.chebyshev_gaussian(P_model.matrix0, features_matrix, args.step, args.mu, args.theta)
    t2 = time.time()
    torch.save(torch.tensor(embeddings_matrix), args.emb2)
    print(f"prone time for adj {args.adj_order}, tebd_dim {args.tebd_dim} is {t2-t1}s")
if __name__ == '__main__':
    main()
