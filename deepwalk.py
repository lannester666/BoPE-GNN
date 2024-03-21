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
    deepwalk = DeepWalk(dgl_graph, emb_dim=args.tebd_dim, walk_length=3, window_size=2)
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
    torch.save(X, f"save_embd/{args.dataset}_deepwalk_{args.tebd_dim}.pt")
    return
    ###################
    for run in range(args.runs):
        split_idx = split_idx_lst[run]
        # split_idx = re_split(dataset.graph, split_idx, args.train_prop, args.valid_prop)
        train_idx = split_idx['train'].to(device)
        model0 = parse_method(args, dataset, n, c, d, device)
        model0.train()
        if args.method == 'gcnt':
            # model0.treduc(dataset, args.dataset)
            model0.treduc_save(dataset, args.dataset)
            return
        ####### dr-gst data ########
        g = dataset.graph
        features = g['node_feat'].clone()
        adj = g['edge_index']
        
        labels = dataset.label.squeeze(1)
        train_index = split_idx['train'].to(device)
        val_idx = split_idx['valid']
        test_idx = split_idx['test']
        n_node = labels.shape[0]
        nclass = labels.max().item() + 1
        mask = torch.zeros(n_node,dtype=bool)
        idx_train = mask.clone().to(device)
        idx_train[train_index] = True
        idx_val = mask.clone().to(device)
        idx_val[val_idx] = True
        idx_test = mask.clone().to(device)
        idx_test[test_idx] = True
        idx_pseudo = torch.zeros_like(idx_train).to(device)
        idx_train_ag = idx_train.clone().to(device)
        train_idx_ag = torch.where(idx_train_ag)[0].to(device)
            
        pseudo_labels = labels.clone().to(device)
        bald = torch.ones(n_node).to(device)
        T = nn.Parameter(torch.eye(nclass, nclass).to(device))
        T.requires_grad = False
        dataset.graph = ori_graph
        
        if args.st_type == 'aug_st':
            if args.aug_type == 'sfp':
                pred = None 
                graph = static_flip_graph(g,pred,args.fpp, device = args.device)
                
        for s in range(args.stage):
            fix_flip = None
            if args.sampling:
                if args.num_layers == 2:
                    sizes = [15, 10]
                elif args.num_layers == 3:
                    sizes = [15, 10, 5]
                train_loader = NeighborSampler(dataset.graph['edge_index'], node_idx=train_idx,
                                        sizes=sizes, batch_size=1024,
                                        shuffle=True, num_workers=12)
                subgraph_loader = NeighborSampler(dataset.graph['edge_index'], node_idx=None, sizes=[-1],
                                                batch_size=4096, shuffle=False,
                                                num_workers=12)
            if args.adam:
                optimizer = torch.optim.Adam(model0.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            elif args.SGD:
                optimizer = torch.optim.SGD(model0.parameters(), lr=args.lr, nesterov=args.nesterov, momentum=args.momentum)
            else:
                optimizer = torch.optim.AdamW(
                    model0.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            best_val = float('-inf')
            test_val = 0
            weight = None
            std_loss = 0
            for epoch in range(args.epochs):
                t1 = time.time()
                model0.train()
                
                if not args.sampling:
                    optimizer.zero_grad()
                    out = model0(dataset)
                    if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
                        if dataset.label.shape[1] == 1:
                            true_label = F.one_hot(pseudo_labels, pseudo_labels.max() + 1).squeeze(1)
                        else:
                            true_label = pseudo_labels

                        loss = criterion(out[train_idx_ag], true_label.squeeze(1)[
                                        train_idx_ag].to(torch.float))
                        
                    else:
                        out = F.log_softmax(out, dim=1)
                        if weight!= None:
                            criterion.weight = weight
                        loss = criterion(
                        out[train_idx_ag], pseudo_labels[train_idx_ag]) 
                    loss.backward()
                    optimizer.step()
                else:
                    for batch_size, n_id, adjs in train_loader:
                        adjs = [adj.to(device) for adj in adjs]
                        optimizer.zero_grad()
                        out = model0(dataset, adjs, dataset.graph['node_feat'][n_id])
                        out = F.log_softmax(out, dim=1)
                        loss = criterion(out, pseudo_labels[n_id[:batch_size]])
                        loss.backward()
                        optimizer.step()
                t2 = time.time()
                result = pseudo_evaluate(model0, dataset, train_index, val_idx, test_idx, eval_func, sampling=args.sampling, subgraph_loader=subgraph_loader)
                t3 = time.time()
                if args.method=='gcnt':
                    temp = class_acc( out[train_idx_ag], dataset.label[train_idx_ag])
                    model0.train_acc = torch.tensor(temp).to(device)
                # print(f"train time: {t2-t1}, eval time: {t3-t2}")
                if result[1] > best_val:
                    best_val = result[1]
                    if args.dataset != 'ogbn-proteins':
                        best_out = F.softmax(result[-1], dim=1)
                    else:
                        best_out = result[-1]
                    best_output = result[-1]
                    test_val = result[2]
                #### test weighted loss
                    # label_acc_list = []
                    # num_label_list = []
                    # mis_label = []
                    # model0.eval()
                    # if not args.sampling:
                    #     out = model0(dataset)
                    # pred = out.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
                    # for l in range(c):
                    #     mis = []
                    #     tmp = dataset.label[split_idx['test']]
                    #     label_cur = torch.where(tmp==l)[0]
                    #     num_label_list.append(label_cur.shape[0])
                    #     pred_l = pred[split_idx['test']][label_cur.detach().cpu()]
                    #     for lab in range(c):
                    #         mis.append(pred_l[pred_l==lab].shape[0])
                    #     mis_label.append(mis)
                    #     try:
                    #         test_acc = eval_func(tmp[label_cur], out[split_idx['test']][label_cur])
                    #     except:
                    #         test_acc = 0
                    #     label_acc_list.append(test_acc)
                    # print(f"epoch {epoch} label_acc_list is {label_acc_list} loss is {loss}")
                    # weight = torch.tensor([(1-x) for x in label_acc_list]).to(device)
                    # std_loss = np.std(label_acc_list)
                logger.add_result(run, result[:-1])

                if epoch % args.display_step ==0:
                    print('Run {}'.format(run)+ ' Stage {}'.format(s) + f' |' +
                                   f'| Val: {100*best_val:.2f} | Test: {100*test_val:.2f}')
                    print(f'Train: {100 * result[0]:.2f}%, ')
            print('Run {}'.format(run)+ ' Stage {}'.format(s) + f' |' +
                                    f'| Val: {100*best_val:.2f} | Test: {100*test_val:.2f}')
            train_acc_list = class_acc(best_output[train_idx_ag], dataset.label[train_idx_ag])
            print('Training Recall of each class is ', train_acc_list)
            val_acc_list = class_acc(best_output[val_idx], dataset.label[val_idx])
            print('Validation Recall of each class is ', val_acc_list)
            test_acc_list = class_acc(best_output[test_idx], dataset.label[test_idx])
            print('Test Recall of each class is ', val_acc_list)
            
            ######  self-training to find pesudo label  ########
            if 'st' in args.st_type:
                if args.st_type == 'aug_st':
                    if args.aug_type == 'rfp':
                        flip_prob = update_flip_prob(flip_prob, best_output, g,beta=args.beta)
        
                    elif args.aug_type == 'sfp':
                        graph = static_flip_graph(g, best_output, args.fpp, device = args.device)
                        dataset.graph['edge_weight'] = graph['edge_weight']
                        model0(dataset)
                        
                idx_unlabeled = ~(idx_train | idx_test | idx_val)
                idx_train_ag, pseudo_labels, idx_pseudo = regenerate_pseudo_label(best_output, labels, idx_train, train_idx_ag, idx_unlabeled,
                                                                                args.threshold,device,nclass)
                train_idx_ag = torch.where(idx_train_ag)[0].to(device)

             
    ### Save results ###
    best_val, best_test = logger.print_statistics()
    filename = f'results/{args.dataset}.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
        write_obj.write(f"{args.method}," + f"{sub_dataset}" +
                        f"{best_val.mean():.3f} ± {best_val.std():.3f}," +
                        f"{best_test.mean():.3f} ± {best_test.std():.3f}\n")

    print(f"Test:{best_test},{args.method}_{args.dataset}_tebd_dim_{args.tebd_dim}_adj_order_{args.adj_order}_layers_{args.num_layers}_hidden_dim_{args.hidden_channels}_w1_{args.w1}_tdropoout_{args.tdropout}_tebd_type_{args.tebd_type}") 
if __name__ == '__main__':
    main()
