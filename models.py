import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SGConv, GATConv, JumpingKnowledge, APPNP, GCN2Conv, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
import scipy.sparse
from tqdm import tqdm
import dgl
from dgl import function as fn
from scipy.sparse import csr_matrix, diags, identity, csgraph, hstack,csc_array,csr_array, eye ,coo_matrix 
import time
from data_utils import get_memory_usage
import random
class LINKX(nn.Module):	
    """ our LINKX method with skip connections 
        a = MLP_1(A), x = MLP_2(X), MLP_3(sigma(W_1[a, x] + a + x))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=False, inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1):
        super(LINKX, self).__init__()	
        self.mlpA = MLP(num_nodes, hidden_channels, hidden_channels, init_layers_A, dropout=0)
        self.mlpX = MLP(in_channels, hidden_channels, hidden_channels, init_layers_X, dropout=0)
        self.W = nn.Linear(2*hidden_channels, hidden_channels)
        self.mlp_final = MLP(hidden_channels, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.A = None
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout

    def reset_parameters(self):	
        self.mlpA.reset_parameters()	
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()	

    def forward(self, data):	
        m = data.graph['num_nodes']	
        feat_dim = data.graph['node_feat']	
        row, col = data.graph['edge_index']	
        row = row-row.min()
        A = SparseTensor(row=row, col=col,	
                 sparse_sizes=(m, self.num_nodes)
                        ).to_torch_sparse_coo_tensor()
        # A = -A
        xA = self.mlpA(A, input_tensor=True)
        xX = self.mlpX(data.graph['node_feat'], input_tensor=True)
        x = torch.cat((xA, xX), axis=-1)
        x = self.W(x)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)
        x = F.relu(x + xA + xX)
        x = self.mlp_final(x, input_tensor=True)

        return x


class LINKX_pc(nn.Module):	
    """ our LINKX method with skip connections 
        a = MLP_1(A), x = MLP_2(X), MLP_3(sigma(W_1[a, x] + a + x))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=False, inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1,
                 tebd_dim = 512, adj_order=1, tdropout = 0.1, w1=1., input_dropout = 0.5, plus = True):
        super(LINKX_pc, self).__init__()	
        self.tebd_dim = tebd_dim
        self.adj_order = adj_order
        self.tdropout = tdropout
        self.input_dropout = input_dropout
        self.plus = plus
        self.mlpA = MLP(num_nodes, hidden_channels, hidden_channels, init_layers_A, dropout=0)
        self.mlpX = MLP(in_channels, hidden_channels, hidden_channels, init_layers_X, dropout=0)
        self.W = nn.Linear(2*hidden_channels, hidden_channels)
        self.mlp_final = MLP(hidden_channels, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.dropout = dropout
        # self.linear = nn.Linear(tebd_dim*adj_order, out_channels)
        self.W1 = nn.Linear(tebd_dim*adj_order, hidden_channels)
        self.W2 = nn.Linear(hidden_channels, out_channels)
        self.topo = None
        self.w1 = w1
        self.A = None
        self.tdropout = tdropout
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout
        self.train_acc = torch.zeros([out_channels])+1e-10 
    def reset_parameters(self):	
        self.mlpA.reset_parameters()	
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()	
        # self.linear.reset_parameters()
        self.W1.reset_parameters()
        self.W2.reset_parameters()
    def forward(self, data, alpha=0.5):	
        if self.topo== None and self.plus :
            combine_list = []
            try:
                A_r_list = torch.load(f"save_embd/{self.data_name}_ours_{self.tebd_dim}_3_A_r_list.pt")[:self.adj_order]
                A_r_pos_list = torch.load(f"save_embd/{self.data_name}_ours_{self.tebd_dim}_3_A_r_list_pos.pt")[:self.adj_order]
            except:
                pass
            for i in range(self.adj_order):
                value1 = A_r_list[i].coalesce().values()
                value2 = A_r_pos_list[i].coalesce().values()
                alpha = torch.mean(value1) / torch.mean(value2)
                combine = SparseTensor(row=A_r_list[i].coalesce().indices()[0], col=A_r_list[i].coalesce().indices()[1], value=value1+alpha*value2, sparse_sizes=(A_r_list[i].shape[0], A_r_list[i].shape[1])).to_torch_sparse_coo_tensor()
                combine_list.append(combine)
            self.topo = torch.cat(combine_list,dim=-1).to(data.graph['node_feat'].device)
        elif self.topo== None and not self.plus:
            A_r_list = torch.load(f"save_embd/{self.data_name}_ours_{self.tebd_dim}_3_A_r_list.pt")[:self.adj_order]
            self.topo = torch.cat(A_r_list,dim=-1).to(data.graph['node_feat'].device)
        m = data.graph['num_nodes']	
        feat_dim = data.graph['node_feat']	
        row, col = data.graph['edge_index']	
        row = row-row.min()
        A = SparseTensor(row=row, col=col,	
                 sparse_sizes=(m, self.num_nodes)
                        ).to_torch_sparse_coo_tensor()
        # A = -A
        x = data.graph['node_feat']
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        xA = self.mlpA(A, input_tensor=True)
        xX = self.mlpX(x, input_tensor=True)
        x = torch.cat((xA, xX), axis=-1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.W(x)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)
        x = F.relu(x + xA + xX)
        x = self.mlp_final(x, input_tensor=True)
        t = self.W1(self.topo)      
        t = F.relu(t)
        t = F.dropout(t, p=self.tdropout, training=self.training)
        t = self.W2(t)
        x_softmax = F.softmax(x,dim=-1).detach()
        t_softmax = F.softmax(t,dim=-1).detach()
        
        x_label = torch.argmax(x_softmax,dim=-1)
        t_label = torch.argmax(t_softmax,dim=-1)
        
        x_conf = torch.max(x_softmax,dim=-1).values
        t_conf = torch.max(t_softmax,dim=-1).values
        
        x_acc = self.train_acc[x_label].to(x_conf.device)
        t_acc = self.train_acc[t_label].to(x_conf.device)
        # print(torch.min(x_acc), torch.min(t_acc), torch.min(x_conf), torch.min(t_conf))
        # r = (self.w1 * x_acc*t_acc/(x_conf*t_conf)).resize(self.num_nodes,1)
        
        r = (self.w1 * (t_acc*t_conf)/(x_acc*x_conf+1e-9)).resize(self.num_nodes,1)

        out  = x * 1/(r+1.) + t * r/(r+1.) 
        return out

class LINKX_pc_plus(nn.Module):	
    """ our LINKX method with skip connections 
        a = MLP_1(A), x = MLP_2(X), MLP_3(sigma(W_1[a, x] + a + x))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=False, inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1,
                 tebd_dim = 512, adj_order=1, tdropout = 0.1, w1=1., input_dropout=0.5, plus=True):
        super(LINKX_pc, self).__init__()	
        self.tebd_dim = tebd_dim
        self.adj_order = adj_order
        self.tdropout = tdropout
        self.input_dropout = input_dropout
        self.plus = plus
        self.mlpA = MLP(num_nodes, hidden_channels, hidden_channels, init_layers_A, dropout=0)
        self.mlpX = MLP(in_channels+tebd_dim*adj_order, hidden_channels, hidden_channels, init_layers_X, dropout=0)
        self.W = nn.Linear(2*hidden_channels, hidden_channels)
        self.mlp_final = MLP(hidden_channels, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.dropout = dropout
        # self.linear = nn.Linear(tebd_dim*adj_order, out_channels)
        # self.W1 = nn.Linear(tebd_dim*adj_order, hidden_channels)
        # self.W2 = nn.Linear(hidden_channels, out_channels)
        self.topo = None
        self.w1 = w1
        self.A = None
        self.tdropout = tdropout
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout
        self.train_acc = torch.zeros([out_channels])+1e-10 
    def reset_parameters(self):	
        self.mlpA.reset_parameters()	
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()	
        # self.linear.reset_parameters()
        # self.W1.reset_parameters()
        # self.W2.reset_parameters()
    def forward(self, data, alpha=0.5):	
        if self.topo== None:
            combine_list = []
            try:
                A_r_list = torch.load(f"save_embd/{self.data_name}_ours_{self.tebd_dim}_3_A_r_list_rand.pt")[:self.adj_order]
                A_r_pos_list = torch.load(f"save_embd/{self.data_name}_ours_{self.tebd_dim}_3_A_r_list_pos.pt")[:self.adj_order]
            except:
                pass
            for i in range(self.adj_order):
                value1 = A_r_list[i].coalesce().values()
                value2 = A_r_pos_list[i].coalesce().values()
                alpha = torch.mean(value1) / torch.mean(value2)
                combine = SparseTensor(row=A_r_list[i].coalesce().indices()[0], col=A_r_list[i].coalesce().indices()[1], value=value1+alpha*value2, sparse_sizes=(A_r_list[i].shape[0], A_r_list[i].shape[1])).to_torch_sparse_coo_tensor()
                combine_list.append(combine)
            self.topo = torch.cat(combine_list,dim=-1).to(data.graph['node_feat'].device)
        m = data.graph['num_nodes']	
        feat_dim = data.graph['node_feat']	
        row, col = data.graph['edge_index']	
        row = row-row.min()
        A = SparseTensor(row=row, col=col,	
                 sparse_sizes=(m, self.num_nodes)
                        ).to_torch_sparse_coo_tensor()
        # A = -A
        A = torch.cat(A)
        xA = self.mlpA(A, input_tensor=True)
        xX = self.mlpX(data.graph['node_feat'], input_tensor=True)
        x = torch.cat((xA, xX), axis=-1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.W(x)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)
        x = F.relu(x + xA + xX)
        x = self.mlp_final(x, input_tensor=True)
        t = self.W1(self.topo)      
        t = F.relu(t)
        t = F.dropout(t, p=self.tdropout, training=self.training)
        t = self.W2(t)
        x_softmax = F.softmax(x,dim=-1).detach()
        t_softmax = F.softmax(t,dim=-1).detach()
        
        x_label = torch.argmax(x_softmax,dim=-1)
        t_label = torch.argmax(t_softmax,dim=-1)
        
        x_conf = torch.max(x_softmax,dim=-1).values
        t_conf = torch.max(t_softmax,dim=-1).values
        
        x_acc = self.train_acc[x_label].to(x_conf.device)
        t_acc = self.train_acc[t_label].to(x_conf.device)
        # print(torch.min(x_acc), torch.min(t_acc), torch.min(x_conf), torch.min(t_conf))
        # r = (self.w1 * x_acc*t_acc/(x_conf*t_conf)).resize(self.num_nodes,1)
        
        r = (self.w1 * (t_acc*t_conf)/(x_acc*x_conf+1e-9)).resize(self.num_nodes,1)

        out  = x * 1/(r+1.) + t * r/(r+1.) 
        return out

class LINK(nn.Module):
    """ logistic regression on adjacency matrix """
    
    def __init__(self, num_nodes,out_channels, order=1):
        super(LINK, self).__init__()
        self.W = nn.Linear(num_nodes, out_channels)
        self.order = order
        self.num_nodes = num_nodes
        self.A = None
    def reset_parameters(self):
        self.W.reset_parameters()
        
    def forward(self, data, find_each_layer = False):
        N = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        if self.A == None:
            if isinstance(edge_index, torch.Tensor):
                row, col = edge_index
                row = row-row.min() # for sampling
                A = SparseTensor(row=row, col=col, sparse_sizes=(N, self.num_nodes)).to_torch_sparse_coo_tensor()
            elif isinstance(edge_index, SparseTensor):
                A = edge_index.to_torch_sparse_coo_tensor()
            for i in range(self.order-1):
                A = A.detach().cpu() @ A.detach().cpu()
            self.A = A.to(edge_index.device)
        logits = self.W(self.A)
        if find_each_layer:
            return logits, []
        else:
            return logits

class LINK_adj(nn.Module):
    """ logistic regression on adjacency matrix """
    
    def __init__(self, num_nodes, out_channels, adj_order = 1):
        super(LINK_adj, self).__init__()
        self.W = nn.Linear(num_nodes*adj_order, out_channels)
        self.num_nodes = num_nodes
        self.adj_order = adj_order
        self.A1 = None
        self.data_name = None
    def reset_parameters(self):
        self.W.reset_parameters()
    
    def save_adj(self, data, data_name, deg_border = 100):
        self.data_name = data_name
        N = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            row = row-row.min() # for sampling
            A = SparseTensor(row=row, col=col, sparse_sizes=(N, self.num_nodes)).to_torch_sparse_coo_tensor()
        elif isinstance(edge_index, SparseTensor):
            A = edge_index.to_torch_sparse_coo_tensor()
        A_r_list = []
        A_r_list.append(A)
        A1 = A
        for i in range(1, self.adj_order):
            A1 = A1 @ A
            srcs = []
            dsts = []
            tmp_graph = dgl.graph((A1._indices()[0], A1._indices()[1]), num_nodes=N)
            tmp_graph.edata['w'] = A1._values()
            for node in range(N):
                deg = tmp_graph.in_degrees(node)
                if deg > deg_border:
                    src, dst = tmp_graph.in_edges(node)
                    idxs = random.sample(range(len(src)), deg_border )
                    src_insert = src[idxs]
                    dst_insert = dst[idxs]
                    srcs.append(src_insert)
                    dsts.append(dst_insert)
            A1 = SparseTensor(row=torch.cat(srcs), col=torch.cat(dsts), sparse_sizes=(N, N)).to_torch_sparse_coo_tensor()

                    
                    
            A_r_list.append(A1)
        
        # self.A_r_list = A_r_list
        self.A1 = torch.cat(A_r_list,dim=-1)
        torch.save(A_r_list, f"save_embd/{data_name}_{self.adj_order}.pt")
            # import pdb; pdb.set_trace()
            # A_r_list = torch.load(f"save_embd/snap-patents_ours_256_2_A_r_list.pt")
            
    def forward(self, data, find_each_layer = False):
        if find_each_layer:
            out_list = []
        if self.A1 == None:
            try:
                A_r_list = torch.load(f"save_embd/{self.data_name}_3.pt")
            except:
                self.save_adj(data, self.data_name)
                A_r_list = torch.load(f"save_embd/{self.data_name}_3.pt")
            # import pdb; pdb.set_trace()
            # A_r_list = torch.load(f"save_embd/snap-patents_ours_256_2_A_r_list.pt")
            A_r_list = A_r_list[:self.adj_order]
            A1 = torch.cat(A_r_list, dim=-1)
            self.A1 = A1.to(data.graph['edge_index'].device)
        if find_each_layer:
            out_list.append(self.A1)
        logits = self.W(self.A1)
        if find_each_layer:
            return logits, out_list
        else:
            return logits
class PCNet(nn.Module):
    def __init__(self, in_channels, out_channels,hidden_channels=256, adj_order = 1, dropout=0.5, input_dropout=0.5):
        super(PCNet, self).__init__()
        self.adj_order = adj_order
        self.tebd_dim = in_channels
        self.W = nn.Linear(in_channels*adj_order, hidden_channels)
        self.W2 = nn.Linear(hidden_channels, out_channels)
        self.input_dropout = input_dropout
        # self.W = nn.Linear(in_channels*adj_order, out_channels)
        self.A1 = None
        self.data_name = None
        self.dropout = dropout
    def reset_parameters(self):
        self.W.reset_parameters()
        self.W2.reset_parameters()
    def save_adj(self, data, data_name):
        self.data_name = data_name
        N = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            row = row-row.min() # for sampling
            A = SparseTensor(row=row, col=col, sparse_sizes=(N, N)).to_torch_sparse_coo_tensor()
        elif isinstance(edge_index, SparseTensor):
            A = edge_index.to_torch_sparse_coo_tensor()
        A_o_list = []
        A = A.detach().cpu()
        A_o_list.append(A)
        A1 = A
        for i in range(1, self.adj_order):
            A1 = A1 @ A
            A_o_list.append(A1)
        nnodes = N
        row = [i for i in range(nnodes)] 
        group_length = int(nnodes/self.tebd_dim)+1
        col = [ (int(i/ group_length))  for i in range(nnodes)]
        reduction_matrix = torch.tensor([row, col])
        row, col = reduction_matrix
        
        reduction_matrix_sp = SparseTensor(   row=row, col=col, sparse_sizes=(nnodes, self.tebd_dim)).to_torch_sparse_coo_tensor()
        
        A_r_list = []
        for A_o  in A_o_list:
            A_r = A_o @ reduction_matrix_sp 
            A_r_list.append(A_r) 
        
        # self.A_r_list = A_r_list
        self.A1 = torch.cat(A_r_list,dim=-1).to(data.graph['edge_index'].device)
        torch.save(A_r_list, f"save_embd/{data_name}_{self.adj_order}.pt")
    def forward(self, data, dataname = None, find_each_layer = False):
        # self.save_adj(data, self.data_name)
        out_list = []
        if self.A1 == None:
            A_r_list = torch.load(f"save_embd/{self.data_name}_ours_{self.tebd_dim}_3_A_r_list.pt")[:self.adj_order]
            self.A1 = torch.cat(A_r_list, dim=-1).to(data.graph['node_feat'].device)
            
        # if self.A1== None:
        #     combine_list = []
        #     try:
        #         A_r_list = torch.load(f"save_embd/{self.data_name}_ours_{self.tebd_dim}_3_A_r_list.pt")[:self.adj_order]
        #         A_r_pos_list = torch.load(f"save_embd/{self.data_name}_ours_{self.tebd_dim}_3_A_r_list_pos.pt")[:self.adj_order]
        #     except:
        #         pass
        #     for i in range(self.adj_order):
        #         value1 = A_r_list[i].coalesce().values()
        #         value2 = A_r_pos_list[i].coalesce().values()
        #         alpha = torch.mean(value1) / torch.mean(value2)
        #         combine = SparseTensor(row=A_r_list[i].coalesce().indices()[0], col=A_r_list[i].coalesce().indices()[1], value=value1+alpha*value2, sparse_sizes=(A_r_list[i].shape[0], A_r_list[i].shape[1])).to_torch_sparse_coo_tensor()
        #         combine_list.append(combine)
        #     self.A1 = torch.cat(combine_list,dim=-1).to(data.graph['node_feat'].device)
        # self.A1 = F.dropout(self.A1, p=self.input_dropout, training=self.training)
        logits = self.W(self.A1)
        logits = F.dropout(logits, p=self.dropout, training=self.training)
        # logits = F.elu(logits)
        logits = F.relu(logits)
        logits = self.W2(logits)
        if find_each_layer:
            return logits, out_list
        else:
            return logits
            


class LINK_Concat(nn.Module):	
    """ concate A and X as joint embeddings i.e. MLP([A;X])"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=True):	
        super(LINK_Concat, self).__init__()	
        self.mlp = MLP(in_channels + num_nodes, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels	
        self.cache = cache
        self.x = None

    def reset_parameters(self):	
        self.mlp.reset_parameters()	

    def forward(self, data):	
        if (not self.cache) or (not isinstance(self.x, torch.Tensor)):
                N = data.graph['num_nodes']	
                feat_dim = data.graph['node_feat']	
                row, col = data.graph['edge_index']	
                col = col + self.in_channels	
                feat_nz = data.graph['node_feat'].nonzero(as_tuple=True)	
                feat_row, feat_col = feat_nz	
                full_row = torch.cat((feat_row, row))	
                full_col = torch.cat((feat_col, col))	
                value = data.graph['node_feat'][feat_nz]	
                full_value = torch.cat((value, 	
                                torch.ones(row.shape[0], device=value.device)))	
                x = SparseTensor(row=full_row, col=full_col,	
                         sparse_sizes=(N, N+self.in_channels)	
                            ).to_torch_sparse_coo_tensor()	
                if self.cache:
                    self.x = x
        else:
                x = self.x
        logits = self.mlp(x, input_tensor=True)
        return logits


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.alpha = 0.2
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False, find_each_layer=False):
        
        if find_each_layer:
            out_list = []
        if not input_tensor:
            x = data.graph['node_feat']
        else:
            x = data
        ori_x = data
        for i, lin in enumerate(self.lins[:-1]):
            prev_x = x
            x =  (1-self.alpha)*lin(x) + self.alpha*ori_x
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # if i>0:
            #     x = x+prev_x
            if find_each_layer:
                out_list.append(x)
        # import pdb; pdb.set_trace()
        x = self.lins[-1](x)
        if find_each_layer:
            return x, out_list
        else:
            return x
    
    
class SGC(nn.Module):
    def __init__(self, in_channels, out_channels, hops):
        """ takes 'hops' power of the normalized adjacency"""
        super(SGC, self).__init__()
        self.conv = SGConv(in_channels, out_channels, hops, cached=True) 

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = data.graph['node_feat']
        if 'edge_weight' in data.graph:
            x = self.conv(x, edge_index, data['edge_weight'])
        else:
            x = self.conv(x, edge_index)
        return x


class SGCMem(nn.Module):
    def __init__(self, in_channels, out_channels, hops):
        """ lower memory version (if out_channels < in_channels)
        takes weight multiplication first, then propagate
        takes hops power of the normalized adjacency
        """
        super(SGCMem, self).__init__()

        self.lin = nn.Linear(in_channels, out_channels)
        self.hops = hops

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = data.graph['node_feat']
        x = self.lin(x)
        n = data.graph['num_nodes']
        edge_weight=None

        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False,
                 dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            edge_weight=None
            adj_t = edge_index

        for _ in range(self.hops):
            x = matmul(adj_t, x)
        
        return x


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=False, use_bn=True):
        super(GCN, self).__init__()

        cached = False
        add_self_loops = True
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        for i, conv in enumerate(self.convs[:-1]):
            if 'edge_weight' in data.graph:
                x = conv(x, data.graph['edge_index'], data.graph['edge_weight'])
            else:
                x = conv(x, data.graph['edge_index'])
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if 'edge_weight' in data.graph:
                x = self.convs[-1](x, data.graph['edge_index'], data.graph['edge_weight'])
        else:
                x = self.convs[-1](x, data.graph['edge_index'])
        return x
    # def forward(self, data, find_each_layer=False):
    #     if find_each_layer:
    #         out_list = []
    #     x = data.graph['node_feat']
    #     for i, conv in enumerate(self.convs[:-1]):
    #         last_x = x 
    #         if 'edge_weight' in data.graph:
    #             x = conv(x, data.graph['edge_index'], data.graph['edge_weight'])
    #         else:
    #             x = conv(x, data.graph['edge_index'])
    #         if self.use_bn:
    #             x = self.bns[i](x)
    #         x = self.activation(x)
    #         x = F.dropout(x, p=self.dropout, training=self.training)
    #         if i >0:
    #             x = x+last_x
    #         if find_each_layer:
    #             out_list.append(x)
    #     if 'edge_weight' in data.graph:
    #             x = self.convs[-1](x, data.graph['edge_index'], data.graph['edge_weight'])
    #     else:
    #             x = self.convs[-1](x, data.graph['edge_index'])
    #     if find_each_layer:
    #         return x, out_list
    #     else:
    #         return x


class LINK_with_dr(nn.Module):
    """ logistic regression on adjacency matrix, make dimension reduction for adjacency matrix """
    
    def __init__(self, in_channels,hidden_dim, out_channels, dropout = 0.5, num_layers = 2):
        super(LINK_with_dr, self).__init__()
        self.W1 = nn.Linear(in_channels, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, out_channels)
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.lins.append(nn.Linear(hidden_dim, out_channels))
        self.in_channels = in_channels
        self.adj = None
        self.dropout = dropout
    def reset_parameters(self):
        self.W1.reset_parameters()
        self.W2.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
    def forward(self, data):
        N = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        if self.adj == None:
            adj = torch.sparse_coo_tensor(indices=data.graph['edge_index'], 
                                        values=torch.ones_like(data.graph['edge_index'][0]),
                                        size=(data.graph['num_nodes'], data.graph['num_nodes']), 
                                        device=data.graph['edge_weight'].device,
                                        dtype=torch.float).to(edge_index.device)
            # adj = F.normalize(adj, dim=1)
            U, S, V = torch.svd_lowrank(adj@adj+adj, q=self.in_channels, )
            adj = torch.mm(U, torch.diag(S))
            
            # adj = F.normalize(adj, dim=0)
            self.adj = adj
        hid = self.W1(self.adj)
        hid = F.dropout(hid, )
        hid = F.relu(hid)
        logits = self.W2(hid)
        return logits
class GCNT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_nodes , adj_order, tebd_type, tebd_dim,
                 tdropout=0.1, 
                 num_layers=2, dropout=0.5, save_mem=False, use_bn=True,
                 w1 = 1.,   input_dropout= 0.5, plus=True, 
                 ):
        super(GCNT, self).__init__()

        cached = False
        add_self_loops = True
        self.input_dropout = input_dropout
        self.plus = plus
        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
        self.mlps.append(MLP(hidden_channels, hidden_channels, hidden_channels, num_layers=3))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.mlps.append(MLP(hidden_channels, hidden_channels, hidden_channels, num_layers=3))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        
        self.num_nodes = num_nodes
        self.tebd_type = tebd_type
        self.tebd_dim = tebd_dim 
        
        if tebd_type == 'ori':
            self.W = nn.Linear(num_nodes, out_channels)
        else:
            # self.W = nn.Linear(tebd_dim*adj_order, out_channels) 
            self.W1 = nn.Linear(tebd_dim*adj_order, hidden_channels)
            self.W2 = nn.Linear(hidden_channels, out_channels)
            self.in_channels = in_channels
            self.adj = None
            self.dropout = dropout
            
        self.w1 = w1
        self.train_acc = torch.zeros([out_channels])+1e-10 
        self.adj_order =adj_order
        self.tdropout= tdropout
        self.A_r = None
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        
        self.W1.reset_parameters()
        self.W2.reset_parameters()
    def treduc_save(self, data, dataset, deg_border=100):
        nnodes = self.num_nodes
        edge_index = data.graph['edge_index']
        N = nnodes
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            row = row-row.min() # for sampling
            A = SparseTensor(value = torch.tensor([1. for i in range(row.shape[0])]).to(edge_index.device), row=row, col=col, sparse_sizes=(self.num_nodes, self.num_nodes))
            pos_A = SparseTensor(row=row, col=col,value=col.to(torch.float), sparse_sizes=(N, N)).to_torch_sparse_coo_tensor()
            A1 = SparseTensor(value = torch.tensor([1. for i in range(row.shape[0])]).to(edge_index.device), row=row, col=col, sparse_sizes=(self.num_nodes, self.num_nodes)).to_torch_sparse_coo_tensor()
        elif isinstance(edge_index, SparseTensor):
            A = edge_index.to_torch_sparse_coo_tensor()
        A1 = A1.detach().cpu()
        # pos_A
        
        values = torch.rand((1,N))
        row = torch.tensor([i for i in range(N)])
        col = torch.tensor([i for i in range(N)])
        values = SparseTensor(value = values.squeeze(), row = row, col = col, sparse_sizes=(N, N)).to_torch_sparse_coo_tensor()
        pos_A = pos_A.detach().cpu()
        A_o = A1
        # A_o = A1 @ values
        self.A_o_list = []
        self.A_o_pos_list = []
        self.A_o_list.append(A_o)
        self.A_o_pos_list.append(pos_A)
        t1 = time.time()
        """
        transform = dgl.DropEdge(0.5)
        tmp_graph = dgl.graph((A_o._indices()[0], A_o._indices()[1]), num_nodes=N)
        tmp_graph.edata['w'] = A_o._values()
        tmp_graph = transform(tmp_graph)
        A1 = SparseTensor(row=torch.cat(srcs), col=torch.cat(dsts), sparse_sizes=(N, N)).to_torch_sparse_coo_tensor()
        """
        print(f"{dataset} adj order 1 degree is {A_o._nnz()/A_o.size()[0]}")
        for i in range(1, self.adj_order):
            A_o = A_o @ A1
            row, col = A_o.coalesce().indices()
            A_o_pos = SparseTensor(row=row, col=col,value=col.to(torch.float), sparse_sizes=(N, N)).to_torch_sparse_coo_tensor()
            print(f"{dataset} adj order {i+1} degree is {A_o._nnz()/A_o.size()[0]}")
            # srcs = []
            # dsts = []
            # tmp_graph = dgl.graph((A_o._indices()[0], A_o._indices()[1]), num_nodes=N)
            # tmp_graph.edata['w'] = A_o._values()
            # for node in range(N):
            #     deg = tmp_graph.in_degrees(node)
            #     if deg > deg_border:
            #         src, dst = tmp_graph.in_edges(node)
            #         idxs = random.sample(range(len(src)), deg_border )
            #         src_insert = src[idxs]
            #         dst_insert = dst[idxs]
            #         srcs.append(src_insert)
            #         dsts.append(dst_insert)
            # A_o = SparseTensor(row=torch.cat(srcs), col=torch.cat(dsts), sparse_sizes=(N, N)).to_torch_sparse_coo_tensor()
            self.A_o_list.append(A_o)
            self.A_o_pos_list.append(A_o_pos)
        t2 = time.time()
        # import pdb; pdb.set_trace()
        if self.tebd_type =='ours':
            t3 = time.time()
            row = [i for i in range(nnodes)] 
            group_length = int(nnodes/self.tebd_dim)+1
            col = [ (int(i/ group_length))  for i in range(nnodes)]
            reduction_matrix = torch.tensor([row, col])
            row, col = reduction_matrix
            reduction_matrix_sp = SparseTensor(   row=row, col=col, sparse_sizes=(nnodes, self.tebd_dim)).to_torch_sparse_coo_tensor()
            A_r_pos_list = []
            A_r_list = []
            
            for idx, A_o  in enumerate(self.A_o_list):
                A_o_pos = self.A_o_pos_list[idx]
                A_r = A_o @ reduction_matrix_sp 
                A_r_list.append(A_r) 
                A_r_pos = A_o_pos @ reduction_matrix_sp
                ones = torch.full((1, A_r_pos.shape[0]), 1/ A_r_pos.shape[0]).to(A_r_pos.device).to_sparse()
                normalize_tensor =  ((ones@A_r_pos).to_dense().squeeze())
                for idx in range(normalize_tensor.shape[0]):
                    if normalize_tensor[idx] != 0:
                       normalize_tensor[idx] = 1/normalize_tensor[idx]  
                temp = torch.diag_embed(normalize_tensor).to_sparse()
                A_r_pos = A_r_pos@temp
                A_r_pos_list.append(A_r_pos)
            # self.A_r_list = A_r_list
            self.A_r = torch.cat(A_r_list,dim=-1)
            torch.save(A_r_list, f"save_embd/{dataset}_{self.tebd_type}_{self.tebd_dim}_{self.adj_order}_A_r_list.pt")
            torch.save(A_r_pos_list, f"save_embd/{dataset}_{self.tebd_type}_{self.tebd_dim}_{self.adj_order}_A_r_list_pos.pt")
            self.A_o_list = []
            t4 = time.time()
            print(f"process {dataset} {self.tebd_type} {self.tebd_dim} {self.adj_order} time: {(t2-t1+t4-t3):.2f} s")
        elif self.tebd_type == 'ours_random':
            t3 = time.time()
            row = [i for i in range(nnodes)] 
            group_length = int(nnodes/self.tebd_dim)+1
            col = [ (int(i/ group_length))  for i in range(nnodes)]
            col = np.random.randint(0,self.tebd_dim ,nnodes).tolist()
            reduction_matrix = torch.tensor([row, col])
            row, col = reduction_matrix
            reduction_matrix_sp = SparseTensor(   row=row, col=col, sparse_sizes=(nnodes, self.tebd_dim)).to_torch_sparse_coo_tensor()
            A_r_pos_list = []
            A_r_list = []
            
            for idx, A_o  in enumerate(self.A_o_list):
                A_o_pos = self.A_o_pos_list[idx]
                A_r = A_o @ reduction_matrix_sp 
                A_r_list.append(A_r) 
                A_r_pos = A_o_pos @ reduction_matrix_sp
                ones = torch.full((1, A_r_pos.shape[0]), 1/ A_r_pos.shape[0]).to(A_r_pos.device).to_sparse()
                normalize_tensor =  ((ones@A_r_pos).to_dense().squeeze())
                for idx in range(normalize_tensor.shape[0]):
                    normalize_tensor[idx] += 1e-9
                    normalize_tensor[idx] = 1/normalize_tensor[idx]  
                temp = torch.diag_embed(normalize_tensor).to_sparse()
                A_r_pos = A_r_pos@temp
                A_r_pos_list.append(A_r_pos)
            # self.A_r_list = A_r_list
            self.A_r = torch.cat(A_r_list,dim=-1)
            torch.save(A_r_list, f"save_embd/{dataset}_{self.tebd_type}_{self.tebd_dim}_{self.adj_order}_A_r_list.pt")
            torch.save(A_r_pos_list, f"save_embd/{dataset}_{self.tebd_type}_{self.tebd_dim}_{self.adj_order}_A_r_list_pos.pt")
            self.A_o_list = []
            t4 = time.time()
            print(f"process {dataset} {self.tebd_type} {self.tebd_dim} {self.adj_order} time: {(t2-t1+t4-t3):.2f} s")
        elif self.tebd_type == 'deg_ours':
            t3 = time.time()
            A_r_list = []
            A_ = A
            row1 = torch.tensor([i for i in range(nnodes)]).to(A_o.device)
            group_length = int(nnodes/self.tebd_dim)+1
            
            for A_o  in self.A_o_list:
                deg = A_.sum(dim=0).to(torch.float)
                values, indices = torch.sort(deg)
                v2, i2 = torch.sort(indices)
                
                col1 = torch.tensor([ (int(i/ group_length)) for i in range(nnodes)]).to(A_o.device)
                col1 = col1[i2]
                # col1 = col1[indices]
                
                reduction_matrix_sp = SparseTensor( row=row1, col=col1, sparse_sizes=(nnodes, self.tebd_dim)).to_torch_sparse_coo_tensor()
                 
                A_r = A_o @ reduction_matrix_sp 
                A_r_list.append(A_r)  
                A_ =  A @ A_ 
                
            self.A_r = torch.cat(A_r_list,dim=-1)
            torch.save(A_r_list, f"save_embd/{dataset}_{self.tebd_type}_{self.tebd_dim}_{self.adj_order}_A_r_list.pt")
            self.A_o_list = []
            t4 = time.time()
            print(f"process {dataset} {self.tebd_type} {self.tebd_dim} {self.adj_order} time: {(t2-t1+t4-t3):.2f} s")
        elif self.tebd_type == 'rand_ours':
            t3 = time.time()
            row = [i for i in range(nnodes)] 
            group_length = int(nnodes/self.tebd_dim)+1
            col = np.random.randint(0,self.tebd_dim ,nnodes).tolist()
            import pdb; pdb.set_trace()
            reduction_matrix = torch.tensor([row, col])
            row, col = reduction_matrix
            
            reduction_matrix_sp = SparseTensor( row=row, col=col, sparse_sizes=(nnodes, self.tebd_dim)).to_torch_sparse_coo_tensor()

            A_r_list = []
            for A_o  in self.A_o_list:
                A_r = A_o @ reduction_matrix_sp 
                A_r_list.append(A_r) 
            
            # self.A_r_list = A_r_list
            self.A_r = torch.cat(A_r_list,dim=-1)
            torch.save(A_r_list, f"save_embd/{dataset}_{self.tebd_type}_{self.tebd_dim}_{self.adj_order}_A_r_list.pt")
            self.A_o_list = []
            t4 = time.time()
            print(f"process {dataset} {self.tebd_type} {self.tebd_dim} {self.adj_order} time: {(t2-t1+t4-t3):.2f} s")
        elif self.tebd_type =='tsvd':
            t3 = time.time()
            A_r_list = []
            for A_o  in self.A_o_list:
                U, S, V = torch.svd_lowrank(A_o, q=self.tebd_dim, )
                A_r = torch.mm(U, torch.diag(S))
                A_r_list.append(A_r) 
            self.A_r_list = torch.cat(A_r_list,dim=-1)
            torch.save(A_r_list, f"save_embd/{dataset}_{self.tebd_type}_{self.tebd_dim}_{self.adj_order}_A_r_list.pt")
            self.A_o_list = []
            t4 = time.time()
            print(f"process {dataset} {self.tebd_type} {self.tebd_dim} {self.adj_order} time: {(t2-t1+t4-t3):.2f} s")
        elif self.tebd_type == 'deep_walk':
            pass
        elif self.tebd_type == 'prone':
            pass
        return 
    
    def treduc(self, data, dataset, deg_border=100):
        nnodes = self.num_nodes
        edge_index = data.graph['edge_index']
        N = nnodes
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            row = row-row.min() # for sampling
            A = SparseTensor(value = torch.tensor([1. for i in range(row.shape[0])]).to(edge_index.device), row=row, col=col, sparse_sizes=(self.num_nodes, self.num_nodes))
            A1 = SparseTensor(value = torch.tensor([1. for i in range(row.shape[0])]).to(edge_index.device), row=row, col=col, sparse_sizes=(self.num_nodes, self.num_nodes)).to_torch_sparse_coo_tensor()
        elif isinstance(edge_index, SparseTensor):
            A = edge_index.to_torch_sparse_coo_tensor()
        self.A_o_list = []
        self.A_o_list.append(A1)
        t1 = time.time()
        nedges = A1._nnz()
        transform = dgl.DropEdge(1-nnodes*5/nedges)
        tmp_graph = dgl.graph((A1._indices()[0], A1._indices()[1]), num_nodes=N)
        tmp_graph.edata['w'] = A1._values()
        tmp_graph = transform(tmp_graph)
        A1 = SparseTensor(row=tmp_graph.edges()[0], col=tmp_graph.edges()[1], sparse_sizes=(N, N)).to_torch_sparse_coo_tensor().to(edge_index.device)
        A_o = A1
        self.A_o_list = []
        self.A_o_list.append(A_o)
        t1 = time.time()
        # A1 = SparseTensor(row=torch.cat(srcs), col=torch.cat(dsts), sparse_sizes=(N, N)).to_torch_sparse_coo_tensor()
        for i in range(1, self.adj_order):
            A_o = A_o @ A1
            # srcs = []
            # dsts = []
            # tmp_graph = dgl.graph((A_o._indices()[0], A_o._indices()[1]), num_nodes=N)
            # tmp_graph.edata['w'] = A_o._values()
            # for node in range(N):
            #     deg = tmp_graph.in_degrees(node)
            #     if deg > deg_border:
            #         src, dst = tmp_graph.in_edges(node)
            #         idxs = random.sample(range(len(src)), deg_border )
            #         src_insert = src[idxs]
            #         dst_insert = dst[idxs]
            #         srcs.append(src_insert)
            #         dsts.append(dst_insert)
            # A_o = SparseTensor(row=torch.cat(srcs), col=torch.cat(dsts), sparse_sizes=(N, N)).to_torch_sparse_coo_tensor()
            self.A_o_list.append(A_o)
        t2 = time.time()
        # import pdb; pdb.set_trace()
        if self.tebd_type =='ours':
            t3 = time.time()
            row = [i for i in range(nnodes)] 
            group_length = int(nnodes/self.tebd_dim)+1
            col = [ (int(i/ group_length))  for i in range(nnodes)]
            reduction_matrix = torch.tensor([row, col])
            row, col = reduction_matrix
            
            reduction_matrix_sp = SparseTensor(   row=row, col=col, sparse_sizes=(nnodes, self.tebd_dim)).to_torch_sparse_coo_tensor().to(A_o.device)
            A_r_list = []
            for A_o  in self.A_o_list:
                A_r = A_o @ reduction_matrix_sp 
                A_r_list.append(A_r) 
                
            # self.A_r_list = A_r_list
            self.A_r = torch.cat(A_r_list,dim=-1)
          
            self.A_o_list = []
            t4 = time.time()
            print(f"process {dataset} {self.tebd_type} {self.tebd_dim} {self.adj_order} time: {(t2-t1+t4-t3):.2f} s")
        elif self.tebd_type == 'deg_ours':
            t3 = time.time()
            A_r_list = []
            A_ = A
            row1 = torch.tensor([i for i in range(nnodes)]).to(A_o.device)
            group_length = int(nnodes/self.tebd_dim)+1
            
            for A_o  in self.A_o_list:
                deg = A_.sum(dim=0).to(torch.float)
                values, indices = torch.sort(deg)
                v2, i2 = torch.sort(indices)
                
                col1 = torch.tensor([ (int(i/ group_length)) for i in range(nnodes)]).to(A_o.device)
                col1 = col1[i2]
                # col1 = col1[indices]
                
                reduction_matrix_sp = SparseTensor( row=row1, col=col1, sparse_sizes=(nnodes, self.tebd_dim)).to_torch_sparse_coo_tensor()
                 
                A_r = A_o @ reduction_matrix_sp 
                A_r_list.append(A_r)  
                A_ =  A @ A_ 
                
            self.A_r = torch.cat(A_r_list,dim=-1)
          
            self.A_o_list = []
            t4 = time.time()
            print(f"process {dataset} {self.tebd_type} {self.tebd_dim} {self.adj_order} time: {(t2-t1+t4-t3):.2f} s")
        elif self.tebd_type == 'rand_ours':
            t3 = time.time()
            row = [i for i in range(nnodes)] 
            group_length = int(nnodes/self.tebd_dim)+1
            col = np.random.randint(0,self.tebd_dim ,nnodes).tolist()
            
            reduction_matrix = torch.tensor([row, col])
            row, col = reduction_matrix
            
            reduction_matrix_sp = SparseTensor( row=row, col=col, sparse_sizes=(nnodes, self.tebd_dim)).to_torch_sparse_coo_tensor()
            
            A_r_list = []
            for A_o  in self.A_o_list:
                A_r = A_o @ reduction_matrix_sp 
                A_r_list.append(A_r) 
            
            # self.A_r_list = A_r_list
            self.A_r = torch.cat(A_r_list,dim=-1)
            self.A_o_list = []
            t4 = time.time()
            print(f"process {dataset} {self.tebd_type} {self.tebd_dim} {self.adj_order} time: {(t2-t1+t4-t3):.2f} s")
        elif self.tebd_type =='tsvd':
            t3 = time.time()
            A_r_list = []
            for A_o  in self.A_o_list:
                U, S, V = torch.svd_lowrank(A_o, q=self.tebd_dim, )
                A_r = torch.mm(U, torch.diag(S))
                A_r_list.append(A_r) 
            self.A_r= torch.cat(A_r_list,dim=-1)
         
            self.A_o_list = []
            t4 = time.time()
            print(f"process {dataset} {self.tebd_type} {self.tebd_dim} {self.adj_order} time: {(t2-t1+t4-t3):.2f} s")
        elif self.tebd_type == 'deep_walk':
            pass
        elif self.tebd_type == 'prone':
            pass
        return 

    def forward(self, data, find_each_layer = False, alpha = 1000, test=False):
        if find_each_layer:
            out_list = []
        x = data.graph['node_feat']
        ori_x = x
        if self.A_r == None and self.tebd_type == 'ours' and self.plus==False:
            A_r_list = torch.load(f"save_embd/{self.data_name}_ours_{self.tebd_dim}_3_A_r_list.pt")[:self.adj_order]
            self.A_r = torch.cat(A_r_list,dim=-1).to(x.device)
        elif self.A_r == None and self.tebd_type == 'ours' and self.plus:
                # if self.topo == None:
            combine_list = []
            try:
                A_r_list = torch.load(f"save_embd/{self.data_name}_{self.tebd_type}_{self.tebd_dim}_3_A_r_list_rand.pt")[:self.adj_order]
                A_r_pos_list = torch.load(f"save_embd/{self.data_name}_{self.tebd_type}_{self.tebd_dim}_3_A_r_list_pos.pt")[:self.adj_order]
            except:
                pass
            
            for i in range(self.adj_order):
                # import pdb; pdb.set_trace()
                value1 = A_r_list[i].coalesce().values()
                value2 = A_r_pos_list[i].coalesce().values()
                alpha = torch.mean(value1) / torch.mean(value2)
                combine = SparseTensor(row=A_r_list[i].coalesce().indices()[0], col=A_r_list[i].coalesce().indices()[1], value=value1+alpha*value2, sparse_sizes=(A_r_list[i].shape[0], A_r_list[i].shape[1])).to_torch_sparse_coo_tensor()
                combine_list.append(combine)
            self.A_r = torch.cat(combine_list,dim=-1).to(x.device)
        elif self.tebd_type == 'deepwalk':
            self.A_r = torch.load(f"save_embd/{self.data_name}_{self.tebd_type}_{self.tebd_dim}.pt").to(x.device)
        elif self.tebd_type == 'prone':
            self.A_r = torch.load(f"save_embd/{self.data_name}_{self.tebd_type}_embd_{self.tebd_dim}_enhanced.pt").to(x.device).to(torch.float32)
        if find_each_layer:
            out_list.append(self.A_r)
        self.alpha = 0.2
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            last_x = x 
            if 'edge_weight' in data.graph:
                x = conv(x, data.graph['edge_index'], data.graph['edge_weight'])
            else:
                x = conv(x, data.graph['edge_index'])
            x, mlp_out = self.mlps[i](x, True, True)
            for out in mlp_out:
                x = x + out
            
            if self.use_bn:
                x = self.bns[i](x)
            
            x = self.activation(x)
            x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
            if i >0:
                x = x + last_x
            if find_each_layer:
                out_list.append(x)
        if 'edge_weight' in data.graph:
                x = self.convs[-1](x, data.graph['edge_index'], data.graph['edge_weight'])
        else:
                x = self.convs[-1](x, data.graph['edge_index'])
        
        # for i in range(self.adj_order):
        #     if self.tebd_type !='ori':
        #         A = self.A_r_list[i] 
        #     else:
        #         A = self.A_o_list[i] 
        #     if i ==0:
        #         t = self.W(A)/(i+1)
        #     else:
        #         t = t + self.W(A)/(i+1)

        t = self.W1(self.A_r)
        # t = F.dropout(t, p=self.dropout, training=self.training)        

        t = F.relu(t)
        t = F.dropout(t, p=self.tdropout, training=self.training)
        t = self.W2(t)
        x_softmax = F.softmax(x,dim=-1).detach()
        t_softmax = F.softmax(t,dim=-1).detach()
        
        x_label = torch.argmax(x_softmax,dim=-1)
        t_label = torch.argmax(t_softmax,dim=-1)
        
        x_conf = torch.max(x_softmax,dim=-1).values
        t_conf = torch.max(t_softmax,dim=-1).values
        
        x_acc = self.train_acc[x_label].to(x_conf.device)
        t_acc = self.train_acc[t_label].to(x_conf.device)
        # print(torch.min(x_acc), torch.min(t_acc), torch.min(x_conf), torch.min(t_conf))
        # r = (self.w1 * x_acc*t_acc/(x_conf*t_conf)).resize(self.num_nodes,1)
        
        r = (self.w1 * (t_acc*t_conf)/(x_acc*x_conf+1e-9)).resize(self.num_nodes,1)

        out  = x * 1/(r+1.) + t * r/(r+1.) 
        if find_each_layer:
            out_list.append(out)
            return out, out_list
        elif test:
            return out, r
        else:
            return out
    



class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=2, sampling=False, add_self_loops=True, input_dropout = 00.4):
        super(GAT, self).__init__()

        self.convs = nn.ModuleList()
        # self.convs.append(
        #     GATConv(in_channels, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops))
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                    GATConv(hidden_channels*heads, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops) ) 
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATConv(hidden_channels*heads, out_channels, heads=heads, concat=False, add_self_loops=add_self_loops))
        # self.convs.append(
        #     GATConv(hidden_channels*heads, out_channels, heads=heads, concat=False, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.activation = F.elu 
        self.sampling = sampling
        self.num_layers = num_layers
        self.input_dropout = input_dropout
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, data, adjs=None, x_batch=None):
        import copy
        if not self.sampling:
            x = data.graph['node_feat']
            x = F.dropout(x, p=self.input_dropout, training=self.training)
            for i, conv in enumerate(self.convs[:-1]):
                if 'edge_weight' in data.graph:
                    x = conv(x, data.graph['edge_index'], data.graph['edge_weight'])
                else:
                    x = conv(x, data.graph['edge_index'])
                x = self.bns[i](x)
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            if 'edge_weight' in data.graph:
                    x = self.convs[-1](x, data.graph['edge_index'], data.graph['edge_weight'])
            else:
                    x = self.convs[-1](x, data.graph['edge_index'])
        else:
            x = x_batch
            for i, (edge_index, _, size) in enumerate(adjs):
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = self.bns[i](x)
                    x = self.activation(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x
  

    def inference(self, data, subgraph_loader):
        x_all = data.graph['node_feat'] 
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        total_edges = 0
        device = x_all.device
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = self.bns[i](x)
                    x = self.activation(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all



class GAT_pc(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, w1 = 1.,
                 dropout=0.5, heads=2, sampling=False, add_self_loops=True, adj_order=1,
                 tebd_dim=512, tdropout=0.1, input_dropout = 0.5, plus=True):
        super(GAT_pc, self).__init__()
        self.tebd_dim = tebd_dim
        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops))
        self.plus = plus
        
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                    GATConv(hidden_channels*heads, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops) ) 
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATConv(hidden_channels*heads, out_channels, heads=heads, concat=False, add_self_loops=add_self_loops))
        # self.linear = nn.Linear(tebd_dim*adj_order, out_channels)
        self.W1 = nn.Linear(tebd_dim*adj_order, hidden_channels)
        self.W2 = nn.Linear(hidden_channels, out_channels)
        self.in_channels = in_channels
        self.adj = None
        self.dropout = dropout
        self.activation = F.elu 
        self.sampling = sampling
        self.num_layers = num_layers
        self.adj_order = adj_order
        self.train_acc = torch.zeros([out_channels])+1e-10 
        self.val_acc = torch.zeros([out_channels])+1e-10 
        self.topo = None
        self.w1 = w1
        self.tdropout = tdropout
        self.input_dropout = input_dropout
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        # self.linear.reset_parameters()
        self.W1.reset_parameters()
        self.W2.reset_parameters()

    def forward(self, data, adjs=None, x_batch=None, alpha=0.5, paint=False, find_each_layer=False, plus=True, mode='train'):
        # if self.topo == None:
        #     A_r_list = torch.load(f"save_embd/{self.data_name}_ours_{self.tebd_dim}_5_A_r_list.pt")[:self.adj_order]
        #     self.topo = torch.cat(A_r_list, dim=-1).to(data.graph['node_feat'].device)
        if find_each_layer:
            out_list = []
        
        if self.topo== None and self.plus:
            combine_list = []
            try:
                A_r_list = torch.load(f"save_embd/{self.data_name}_{self.tebd_type}_{self.tebd_dim}_3_A_r_list.pt")[:self.adj_order]
                A_r_pos_list = torch.load(f"save_embd/{self.data_name}_{self.tebd_type}_{self.tebd_dim}_3_A_r_list_pos.pt")[:self.adj_order]
                
            except:
                pass
            for i in range(self.adj_order):
                value1 = A_r_list[i].coalesce().values()
                value2 = A_r_pos_list[i].coalesce().values()
                alpha = torch.mean(value1) / torch.mean(value2)
                # alpha = alpha*0.1
                # import pdb; pdb.set_trace()
                combine = SparseTensor(row=A_r_list[i].coalesce().indices()[0], col=A_r_list[i].coalesce().indices()[1], value=value1+alpha*value2, sparse_sizes=(A_r_list[i].shape[0], A_r_list[i].shape[1])).to_torch_sparse_coo_tensor()
                combine_list.append(combine)
            self.topo = torch.cat(combine_list,dim=-1).to(data.graph['node_feat'].device)
        elif self.topo == None and self.plus==False:
            A_r_list = torch.load(f"save_embd/{self.data_name}_{self.tebd_type}_{self.tebd_dim}_3_A_r_list.pt")[:self.adj_order]
            self.topo = torch.cat(A_r_list, dim=-1).to(data.graph['node_feat'].device)
        self.num_nodes = data.graph['node_feat'].size(0)

        if not self.sampling:
            x = data.graph['node_feat']
            if self.data_name in ['arxiv-year', 'genius']:
                pass
            else:
                x = F.dropout(x, p=self.input_dropout, training=self.training)
            for i, conv in enumerate(self.convs[:-1]):
                if 'edge_weight' in data.graph:
                    x = conv(x, data.graph['edge_index'], data.graph['edge_weight'])
                else:
                    x = conv(x, data.graph['edge_index'])
                x = self.bns[i](x)
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            if 'edge_weight' in data.graph:
                    x = self.convs[-1](x, data.graph['edge_index'], data.graph['edge_weight'])
            else:
                    x = self.convs[-1](x, data.graph['edge_index'])
        else:
            x = x_batch
            for i, (edge_index, _, size) in enumerate(adjs):
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = self.bns[i](x)
                    x = self.activation(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
        # t = self.linear(self.topo)
        # t = F.dropout(t, p=self.tdropout, training=self.training)
        # t = F.dropout(self.topo, p=self.tdropout, training=self.training).to(self.topo.device)
        t = self.W1(self.topo)      

        t = self.activation(t)
        if self.data_name in ['twitch-gamer']:
            pass
        else:
            t = F.dropout(t, p=self.tdropout, training=self.training)
        t = self.W2(t)
        x_softmax = F.softmax(x,dim=-1).detach()
        t_softmax = F.softmax(t,dim=-1).detach()
        
        x_label = torch.argmax(x_softmax,dim=-1)
        t_label = torch.argmax(t_softmax,dim=-1)
        
        x_conf = torch.max(x_softmax,dim=-1).values
        t_conf = torch.max(t_softmax,dim=-1).values
        if mode == 'train':
            x_acc = self.val_acc[x_label].to(x_conf.device)
            t_acc = self.val_acc[t_label].to(x_conf.device)
            # print(torch.min(x_acc), torch.min(t_acc), torch.min(x_conf), torch.min(t_conf))
            # r = (self.w1 * x_acc*t_acc/(x_conf*t_conf)).resize(self.num_nodes,1)
            if paint:
                from data_utils import paint_hist
                paint_hist(self.w1*(t_acc*t_conf)/(x_acc*x_conf+1e-9))
            # r = (self.w1 * (t_acc*t_conf)/(x_acc*x_conf+1e-9)).resize(self.num_nodes,1)
            r=  self.w1*(t_conf/x_conf).resize(self.num_nodes,1)
            out  = x * 1/(r+1.) + t * r/(r+1.) 
            return out
        elif mode == "val":
            x_acc = self.train_acc[x_label].to(x_conf.device)
            t_acc = self.train_acc[t_label].to(x_conf.device)
            m = torch.where(x_acc>t_acc)[0]
            n = torch.where(x_acc<=t_acc)[0]
            out = torch.zeros_like(x)
            out[m] = x[m]
            out[n] = t[n]
            # print(torch.min(x_acc), torch.min(t_acc), torch.min(x_conf), torch.min(t_conf))
            # r = (self.w1 * x_acc*t_acc/(x_conf*t_conf)).resize(self.num_nodes,1)
            # out = 

            # out  = x * 1/(r+1.) + t * r/(r+1.) 
            return out

    def inference(self, data, subgraph_loader):
        x_all = data.graph['node_feat'] 
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        total_edges = 0
        device = x_all.device
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = self.bns[i](x)
                    x = self.activation(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


class MultiLP(nn.Module):
    """ label propagation, with possibly multiple hops of the adjacency """
    
    def __init__(self, out_channels, alpha, hops, num_iters=50, mult_bin=False):
        super(MultiLP, self).__init__()
        self.out_channels = out_channels
        self.alpha = alpha
        self.hops = hops
        self.num_iters = num_iters
        self.mult_bin = mult_bin # handle multiple binary tasks
        
    def forward(self, data, train_idx):
        n = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        edge_weight=None

        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False)
            row, col = edge_index
            # transposed if directed
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False)
            edge_weight=None
            adj_t = edge_index

        y = torch.zeros((n, self.out_channels)).to(adj_t.device())
        if data.label.shape[1] == 1:
            # make one hot
            y[train_idx] = F.one_hot(data.label[train_idx], self.out_channels).squeeze(1).to(y)
        elif self.mult_bin:
            y = torch.zeros((n, 2*self.out_channels)).to(adj_t.device())
            for task in range(data.label.shape[1]):
                y[train_idx, 2*task:2*task+2] = F.one_hot(data.label[train_idx, task], 2).to(y)
        else:
            y[train_idx] = data.label[train_idx].to(y.dtype)
        result = y.clone()
        for _ in range(self.num_iters):
            for _ in range(self.hops):
                result = matmul(adj_t, result)
            result *= self.alpha
            result += (1-self.alpha)*y

        if self.mult_bin:
            output = torch.zeros((n, self.out_channels)).to(result.device)
            for task in range(data.label.shape[1]):
                output[:, task] = result[:, 2*task+1]
            result = output

        return result


class MixHopLayer(nn.Module):
    """ Our MixHop layer """
    def __init__(self, in_channels, out_channels, hops=2):
        super(MixHopLayer, self).__init__()
        self.hops = hops
        self.lins = nn.ModuleList()
        for hop in range(self.hops+1):
            lin = nn.Linear(in_channels, out_channels)
            self.lins.append(lin)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        xs = [self.lins[0](x) ]
        for j in range(1,self.hops+1):
            # less runtime efficient but usually more memory efficient to mult weight matrix first
            x_j = self.lins[j](x)
            for hop in range(j):
                x_j = matmul(adj_t, x_j)
            xs += [x_j]
        return torch.cat(xs, dim=1)

class MixHop(nn.Module):
    """ our implementation of MixHop
    some assumptions: the powers of the adjacency are [0, 1, ..., hops],
        with every power in between
    each concatenated layer has the same dimension --- hidden_channels
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, hops=2):
        super(MixHop, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(MixHopLayer(in_channels, hidden_channels, hops=hops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*(hops+1)))
        for _ in range(num_layers - 2):
            self.convs.append(
                MixHopLayer(hidden_channels*(hops+1), hidden_channels, hops=hops))
            self.bns.append(nn.BatchNorm1d(hidden_channels*(hops+1)))

        self.convs.append(
            MixHopLayer(hidden_channels*(hops+1), out_channels, hops=hops))

        # note: uses linear projection instead of paper's attention output
        self.final_project = nn.Linear(out_channels*(hops+1), out_channels)

        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.final_project.reset_parameters()


    def forward(self, data):
        x = data.graph['node_feat']
        n = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        edge_weight = None
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False,
                 dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            edge_weight=None
            adj_t = edge_index
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        x = self.final_project(x)
        return x

class GCNJK(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=False, jk_type='max'):
        super(GCNJK, self).__init__()

        cached = False
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.jump = JumpingKnowledge(jk_type, channels=hidden_channels, num_layers=1)
        if jk_type == 'cat':
            self.final_project = nn.Linear(hidden_channels * num_layers, out_channels)
        else: # max or lstm
            self.final_project = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph['edge_index'])
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'])
        xs.append(x)
        x = self.jump(xs)
        x = self.final_project(x)
        return x

class GATJK(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=2, jk_type='max'):
        super(GATJK, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 2):

            self.convs.append(
                    GATConv(hidden_channels*heads, hidden_channels, heads=heads, concat=True) ) 
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATConv(hidden_channels*heads, hidden_channels, heads=heads))

        self.dropout = dropout
        self.activation = F.elu # note: uses elu

        self.jump = JumpingKnowledge(jk_type, channels=hidden_channels*heads, num_layers=1)
        if jk_type == 'cat':
            self.final_project = nn.Linear(hidden_channels*heads*num_layers, out_channels)
        else: # max or lstm
            self.final_project = nn.Linear(hidden_channels*heads, out_channels)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()


    def forward(self, data):
        x = data.graph['node_feat']
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph['edge_index'])
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'])
        xs.append(x)
        x = self.jump(xs)
        x = self.final_project(x)
        return x

class H2GCNConv(nn.Module):
    """ Neighborhood aggregation step """
    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)


class H2GCN(nn.Module):
    """ our implementation """
    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, num_nodes,
                    num_layers=2, dropout=0.5, save_mem=False, num_mlp_layers=1,
                    use_bn=True, conv_dropout=True):
        super(H2GCN, self).__init__()

        self.feature_embed = MLP(in_channels, hidden_channels,
                hidden_channels, num_layers=num_mlp_layers, dropout=dropout)


        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv())

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )

        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            if l != num_layers-2:
                self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout # dropout neighborhood aggregation steps

        self.jump = JumpingKnowledge('cat')
        last_dim = hidden_channels*(2**(num_layers+1)-1)
        self.final_project = nn.Linear(last_dim, out_channels)

        self.num_nodes = num_nodes
        self.init_adj(edge_index)

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def init_adj(self, edge_index):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = self.num_nodes
        
        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)
        
        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)



    def forward(self, data):
        x = data.graph['node_feat']
        n = data.graph['num_nodes']

        adj_t = self.adj_t
        adj_t2 = self.adj_t2
        
        x = self.feature_embed(data)
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2) 
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)

        x = self.jump(xs)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_project(x)
        return x



class H2GCN_pc(nn.Module):
    """ our implementation """
    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, num_nodes,w1=1.,
                    num_layers=2, dropout=0.5, save_mem=False, num_mlp_layers=1,
                    use_bn=True, conv_dropout=True,adj_order=1, tebd_dim = 512, tdropout = 0.1, input_dropout=0.5, plus=True):
        super(H2GCN_pc, self).__init__()
        self.tebd_dim = tebd_dim
        self.feature_embed = MLP(in_channels, hidden_channels,
                hidden_channels, num_layers=num_mlp_layers, dropout=dropout)
        self.input_dropout = input_dropout
        self.plus = plus
        self.tdropout = tdropout
        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv())

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )

        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            if l != num_layers-2:
                self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout # dropout neighborhood aggregation steps
        self.train_acc = torch.zeros([out_channels])+1e-10 
        self.jump = JumpingKnowledge('cat')
        last_dim = hidden_channels*(2**(num_layers+1)-1)
        self.final_project = nn.Linear(last_dim, out_channels)
        # self.linear = nn.Linear(adj_order*tebd_dim, out_channels)
        self.W1 = nn.Linear(tebd_dim*adj_order, hidden_channels)
        self.W2 = nn.Linear(hidden_channels, out_channels)
        self.num_nodes = num_nodes
        self.init_adj(edge_index)
        self.topo = None
        self.adj_order = adj_order
        self.w1 = w1
    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        # self.linear.reset_parameters()
        self.W1.reset_parameters()  
        self.W2.reset_parameters()
    def init_adj(self, edge_index):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = self.num_nodes
        
        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)
        
        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)



    def forward(self, data, alpha=0.5):
        if self.topo == None and not self.plus:
            A_r_list = torch.load(f"save_embd/{self.data_name}_ours_{self.tebd_dim}_3_A_r_list.pt")[:self.adj_order]
            self.topo = torch.cat(A_r_list, dim=-1).to(data.graph['node_feat'].device)    
        elif self.topo== None and self.plus:
            combine_list = []
            try:
                A_r_list = torch.load(f"save_embd/{self.data_name}_ours_{self.tebd_dim}_3_A_r_list_rand.pt")[:self.adj_order]
                A_r_pos_list = torch.load(f"save_embd/{self.data_name}_ours_{self.tebd_dim}_3_A_r_list_pos.pt")[:self.adj_order]
            except:
                pass
            for i in range(self.adj_order):
                value1 = A_r_list[i].coalesce().values()
                value2 = A_r_pos_list[i].coalesce().values()
                alpha = torch.mean(value1) / torch.mean(value2)
                combine = SparseTensor(row=A_r_list[i].coalesce().indices()[0], col=A_r_list[i].coalesce().indices()[1], value=value1+alpha*value2, sparse_sizes=(A_r_list[i].shape[0], A_r_list[i].shape[1])).to_torch_sparse_coo_tensor()
                combine_list.append(combine)
            self.topo = torch.cat(combine_list,dim=-1).to(data.graph['node_feat'].device)
        x = data.graph['node_feat']
        n = data.graph['num_nodes']
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        adj_t = self.adj_t
        adj_t2 = self.adj_t2
        
        x = self.feature_embed(data)
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2) 
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x.to_dense())
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x.to_dense())

        x = self.jump(xs)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_project(x)
        t = self.W1(self.topo)      

        t = F.relu(t)
        t = F.dropout(t, p=self.tdropout, training=self.training)
        t = self.W2(t)
        x_softmax = F.softmax(x,dim=-1).detach()
        t_softmax = F.softmax(t,dim=-1).detach()
        
        x_label = torch.argmax(x_softmax,dim=-1)
        t_label = torch.argmax(t_softmax,dim=-1)
        
        x_conf = torch.max(x_softmax,dim=-1).values
        t_conf = torch.max(t_softmax,dim=-1).values
        
        x_acc = self.train_acc[x_label].to(x_conf.device)
        t_acc = self.train_acc[t_label].to(x_conf.device)
        # print(torch.min(x_acc), torch.min(t_acc), torch.min(x_conf), torch.min(t_conf))
        # r = (self.w1 * x_acc*t_acc/(x_conf*t_conf)).resize(self.num_nodes,1)
        
        r = (self.w1 * (t_acc*t_conf)/(x_acc*x_conf+1e-9)).resize(self.num_nodes,1)

        out  = x * 1/(r+1.) + t * r/(r+1.) 
        return out

class APPNP_Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dprate=.0, dropout=.5, K=10, alpha=.1, num_layers=3):
        super(APPNP_Net, self).__init__()
        
        self.mlp = MLP(in_channels, hidden_channels, out_channels, num_layers=num_layers, dropout=dropout)
        self.prop1 = APPNP(K, alpha)

        self.dprate = dprate
        self.dropout = dropout
        
    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = self.mlp(data)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return x





class GPR_prop(MessagePassing):
    '''
    GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, torch.Tensor):
            edge_index, norm = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
            norm = None

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class GPRGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, Init='Random', dprate=.0, dropout=.5, K=10, alpha=.1, Gamma=None, num_layers=3):
        super(GPRGNN, self).__init__()
        
        self.mlp = MLP(in_channels, hidden_channels, out_channels, num_layers=num_layers, dropout=dropout)
        self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout
        
    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = self.mlp(data)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return x


class GCNII(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, alpha, theta, shared_weights=True, dropout=0.5, labels=None):
        super(GCNII, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        
        self.bns = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.labels = labels
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.dropout = dropout
        
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        n = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        if 'edge_weight' in data.graph:
            edge_weight = data.graph['edge_weight']
        else:
            edge_weight = None
        # import pdb; pdb.set_trace()
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False, dtype=x.dtype)
            row, col = edge_index
            # labels = data.label.squeeze(1)
            # edge_weight = torch.where(labels[row] == labels[col], edge_weight, -edge_weight)
    
            # p = torch.rand_like(edge_weight)
            # edge_weight = torch.where(p < 0.5, edge_weight*(-1), edge_weight)
            # ident = torch.where(row == col)[0]
            # edge_weight[ident] = abs(edge_weight[ident])

            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
            
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False, dtype=x.dtype)
            edge_weight=None
            adj_t = edge_index
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            
            # if 'edge_weight' in data.graph:
            #     x = conv(x, x_0, adj_t, edge_weight=data.graph['edge_weight'])
            # else:
            x = conv(x, x_0, adj_t)
            x = self.bns[i](x)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        return x
    

class GCNII_pc(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, alpha, theta, shared_weights=True, dropout=0.5, labels=None,
                 w1=1.,adj_order=1, tebd_dim=512, tdropout=0.1,input_dropout=0.2, plus=True):
        super(GCNII_pc, self).__init__()
        self.tebd_dim = tebd_dim
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.input_dropout = input_dropout
        self.plus = plus
        self.bns = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.labels = labels
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        # self.linear = nn.Linear(tebd_dim*adj_order, out_channels)
        self.W1 = nn.Linear(tebd_dim*adj_order, hidden_channels)
        self.W2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        self.tdropout = tdropout
        self.w1 = w1
        self.adj_order = adj_order
        self.train_acc = torch.zeros([out_channels])+1e-10 
        self.topo = None
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        # self.linear.reset_parameters()
        self.W1.reset_parameters()
        self.W2.reset_parameters()
    def forward(self, data, alpha=0.5):
        
        if self.topo== None and self.plus:
            combine_list = []
            try:
                A_r_list = torch.load(f"save_embd/{self.data_name}_ours_{self.tebd_dim}_3_A_r_list.pt")[:self.adj_order]
                A_r_pos_list = torch.load(f"save_embd/{self.data_name}_ours_{self.tebd_dim}_3_A_r_list_pos.pt")[:self.adj_order]
            except:
                pass
            for i in range(self.adj_order):
                value1 = A_r_list[i].coalesce().values()
                value2 = A_r_pos_list[i].coalesce().values()
                alpha = torch.mean(value1) / torch.mean(value2)
                combine = SparseTensor(row=A_r_list[i].coalesce().indices()[0], col=A_r_list[i].coalesce().indices()[1], value=value1+alpha*value2, sparse_sizes=(A_r_list[i].shape[0], A_r_list[i].shape[1])).to_torch_sparse_coo_tensor()
                combine_list.append(combine)
            self.topo = torch.cat(combine_list,dim=-1).to(data.graph['node_feat'].device)
        elif self.topo== None and not self.plus:
            A_r_list = torch.load(f"save_embd/{self.data_name}_ours_{self.tebd_dim}_3_A_r_list.pt")[:self.adj_order]
            self.topo = torch.cat(A_r_list,dim=-1).to(data.graph['node_feat'].device)
        x = data.graph['node_feat']
     
        n = data.graph['num_nodes']
        self.num_nodes = n
        edge_index = data.graph['edge_index']
        if 'edge_weight' in data.graph:
            edge_weight = data.graph['edge_weight']
        else:
            edge_weight = None
        # import pdb; pdb.set_trace()
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False, dtype=x.dtype)
            row, col = edge_index
            # labels = data.label.squeeze(1)
            # edge_weight = torch.where(labels[row] == labels[col], edge_weight, -edge_weight)
    
            # p = torch.rand_like(edge_weight)
            # edge_weight = torch.where(p < 0.5, edge_weight*(-1), edge_weight)
            # ident = torch.where(row == col)[0]
            # edge_weight[ident] = abs(edge_weight[ident])

            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
            
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False, dtype=x.dtype)
            edge_weight=None
            adj_t = edge_index
        
        x = F.dropout(x, self.input_dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            
            # if 'edge_weight' in data.graph:
            #     x = conv(x, x_0, adj_t, edge_weight=data.graph['edge_weight'])
            # else:
            x = conv(x, x_0, adj_t)
            x = self.bns[i](x)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        # t = self.linear(self.topo)
        # t = F.dropout(t, p=self.tdropout, training=self.training)
        t = self.W1(self.topo)      
        t = F.relu(t)
        t = F.dropout(t, p=self.tdropout, training=self.training)
        t = self.W2(t)
        x_softmax = F.softmax(x,dim=-1).detach()
        t_softmax = F.softmax(t,dim=-1).detach()
        
        x_label = torch.argmax(x_softmax,dim=-1)
        t_label = torch.argmax(t_softmax,dim=-1)
        
        x_conf = torch.max(x_softmax,dim=-1).values
        t_conf = torch.max(t_softmax,dim=-1).values
        
        x_acc = self.train_acc[x_label].to(x_conf.device)
        t_acc = self.train_acc[t_label].to(x_conf.device)
        # print(torch.min(x_acc), torch.min(t_acc), torch.min(x_conf), torch.min(t_conf))
        # r = (self.w1 * x_acc*t_acc/(x_conf*t_conf)).resize(self.num_nodes,1)
        
        r = (self.w1 * (t_acc*t_conf)/(x_acc*x_conf+1e-9)).resize(self.num_nodes,1)

        out  = x * 1/(r+1.) + t * r/(r+1.) 
        return out
class FALayer(nn.Module):
    def __init__(self, g, in_dim, dropout):
        super(FALayer, self).__init__()
        device = g['edge_index'].device
        U = g['edge_index'][0].detach().cpu()
        V = g['edge_index'][1].detach().cpu()

        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        g = dgl.to_bidirected(g)
        g = g.to(device)
        deg = g.in_degrees().cuda().float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        g.ndata['d'] = norm.to(device)
        self.g = g
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        e = g * edges.dst['d'] * edges.src['d']
        e = self.dropout(e)
        return {'e': e, 'm': g}

    def forward(self, h):
        
        self.g.ndata['h'] = h
        self.g.apply_edges(self.edge_applying)
        self.g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))

        return self.g.ndata['z']

class FAGCN(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, dropout, eps, layer_num=2):
        super(FAGCN, self).__init__()
        self.g = g
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(self.g, hidden_dim, dropout))

        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.t2 = nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, data):
        h = data.graph['node_feat']

        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h)
            h = self.eps * raw + h
        h = self.t2(h)
        return F.log_softmax(h, 1)
    
import scipy.sparse
import scipy.sparse as sp
from scipy import linalg
from scipy.special import iv

from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd

class ProNE():
	def __init__(self, graph, emb_file1, emb_file2, dimension):
		self.graph = graph
		self.emb1 = emb_file1
		self.emb2 = emb_file2
		self.dimension = dimension
		self.G = dgl.to_networkx(graph)
	
		self.G = self.G.to_undirected()
		self.node_number = self.G.number_of_nodes()
		matrix0 = scipy.sparse.lil_matrix((self.node_number, self.node_number))

		for e in self.G.edges():
			if e[0] != e[1]:
				matrix0[e[0], e[1]] = 1
				matrix0[e[1], e[0]] = 1
		self.matrix0 = scipy.sparse.csr_matrix(matrix0)
		print(matrix0.shape)

	def get_embedding_rand(self, matrix):
		# Sparse randomized tSVD for fast embedding
		t1 = time.time()
		l = matrix.shape[0]
		smat = scipy.sparse.csc_matrix(matrix)  # convert to sparse CSC format
		print('svd sparse', smat.data.shape[0] * 1.0 / l ** 2)
		U, Sigma, VT = randomized_svd(smat, n_components=self.dimension, n_iter=5, random_state=None)
		U = U * np.sqrt(Sigma)
		U = preprocessing.normalize(U, "l2")
		print('sparsesvd time', time.time() - t1)
		return U

	def get_embedding_dense(self, matrix, dimension):
		# get dense embedding via SVD
		t1 = time.time()
		U, s, Vh = linalg.svd(matrix, full_matrices=False, check_finite=False, overwrite_a=True)
		U = np.array(U)
		U = U[:, :dimension]
		s = s[:dimension]
		s = np.sqrt(s)
		U = U * s
		U = preprocessing.normalize(U, "l2")
		print('densesvd time', time.time() - t1)
		return U

	def pre_factorization(self, tran, mask):
		# Network Embedding as Sparse Matrix Factorization
		t1 = time.time()
		l1 = 0.75
		C1 = preprocessing.normalize(tran, "l1")
		neg = np.array(C1.sum(axis=0))[0] ** l1

		neg = neg / neg.sum()

		neg = scipy.sparse.diags(neg, format="csr")
		neg = mask.dot(neg)
		print("neg", time.time() - t1)

		C1.data[C1.data <= 0] = 1
		neg.data[neg.data <= 0] = 1

		C1.data = np.log(C1.data)
		neg.data = np.log(neg.data)

		C1 -= neg
		F = C1
		features_matrix = self.get_embedding_rand(F)
		return features_matrix

	def chebyshev_gaussian(self, A, a, order=10, mu=0.5, s=0.5):
		# NE Enhancement via Spectral Propagation
		print('Chebyshev Series -----------------')
		t1 = time.time()

		if order == 1:
			return a

		A = sp.eye(self.node_number) + A
		DA = preprocessing.normalize(A, norm='l1')
		L = sp.eye(self.node_number) - DA

		M = L - mu * sp.eye(self.node_number)

		Lx0 = a
		Lx1 = M.dot(a)
		Lx1 = 0.5 * M.dot(Lx1) - a

		conv = iv(0, s) * Lx0
		conv -= 2 * iv(1, s) * Lx1
		for i in range(2, order):
			Lx2 = M.dot(Lx1)
			Lx2 = (M.dot(Lx2) - 2 * Lx1) - Lx0
			#         Lx2 = 2*L.dot(Lx1) - Lx0
			if i % 2 == 0:
				conv += 2 * iv(i, s) * Lx2
			else:
				conv -= 2 * iv(i, s) * Lx2
			Lx0 = Lx1
			Lx1 = Lx2
			del Lx2
			print('Bessell time', i, time.time() - t1)
		mm = A.dot(a - conv)
		emb = self.get_embedding_dense(mm, self.dimension)
		return emb



def save_embedding(emb_file, features):
	# save node embedding into emb_file with word2vec format
	f_emb = open(emb_file, 'w')
	f_emb.write(str(len(features)) + " " + str(features.shape[1]) + "\n")
	for i in range(len(features)):
		s = str(i) + " " + " ".join(str(f) for f in features[i].tolist())
		f_emb.write(s + "\n")
	f_emb.close()
