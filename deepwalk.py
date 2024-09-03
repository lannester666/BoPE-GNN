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
    ### Training loop ###
    ##### deepwalk #####
    dgl_graph = dgl.graph((dataset.graph['edge_index'][0], dataset.graph['edge_index'][1]))
    # deepwalk = DeepWalk(dgl_graph, emb_dim=args.tebd_dim, walk_length=5, window_size=3).to(args.device)
    deepwalk = DeepWalk(dgl_graph, emb_dim=args.tebd_dim, walk_length=5, window_size=3)
    dataloader = DataLoader(torch.arange(dgl_graph.num_nodes()), batch_size=1024, shuffle=True, collate_fn=deepwalk.sample)
    optimizer = SparseAdam(deepwalk.parameters(), lr=0.01)
    num_epochs = 10
    for epoch in range(num_epochs):
        for batch_walks in dataloader:
            loss = deepwalk(batch_walks.detach().cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    X = deepwalk.node_embed.weight.detach()
    # import pdb; pdb.set_trace()
    torch.save(X, f"save_embd/{args.dataset}_deepwalk_{args.tebd_dim}.pt")
  
if __name__ == '__main__':
    main()
