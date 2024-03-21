import torch.nn as nn 
from models import LINK, GCN, MLP, SGC, GAT,PCNet, SGCMem, MultiLP, MixHop, GCNJK, GATJK, H2GCN, APPNP_Net,LINK_Concat, LINKX, GPRGNN, GCNII, FAGCN,GCNT
from data_utils import normalize
import torch 
from models import GAT_pc, GCNII_pc, H2GCN_pc, LINKX_pc, LINK_adj

def parse_method(args, dataset, n, c, d, device):
    if args.method == 'link':
        model = LINK(n, c).to(device)
    elif args.method == 'link_adj':
        model = LINK_adj(n, c, args.adj_order).to(device)
    elif args.method == 'pcnet':
        model = PCNet(args.tebd_dim, c, adj_order=args.adj_order, input_dropout=args.input_dropout).to(device)
    elif args.method == 'gcn':
        if args.dataset == 'ogbn-proteins':
            # Pre-compute GCN normalization.
            dataset.graph['edge_index'] = normalize(dataset.graph['edge_index'])
            model = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        dropout=args.dropout,
                        save_mem=True,
                        use_bn=not args.no_bn).to(device)
        else:
            model = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=not args.no_bn).to(device)
  
    elif args.method == 'gcnt':
        model = GCNT(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_nodes = dataset.graph['num_nodes'], adj_order=args.adj_order, tebd_type=args.tebd_type, tebd_dim=args.tebd_dim,
                        tdropout=args.tdropout,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=not args.no_bn,
                        w1=args.w1 , input_dropout= args.input_dropout, plus=args.plus
                        ).to(device)
        
    
    elif args.method == 'fagcn':
        model = FAGCN(dataset.graph, d, args.hidden_channels, c, dropout=args.dropout, eps=args.eps, layer_num = args.num_layers)
    elif args.method == 'mlp' or args.method == 'cs':
        model = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                    out_channels=c, num_layers=args.num_layers,
                    dropout=args.dropout).to(device)
    elif args.method == 'sgc':
        if args.cached:
            model = SGC(in_channels=d, out_channels=c, hops=args.hops).to(device)
        else:
            model = SGCMem(in_channels=d, out_channels=c,
                           hops=args.hops).to(device)
    elif args.method == 'gprgnn':
        model = GPRGNN(d, args.hidden_channels, c, alpha=args.gpr_alpha, num_layers=args.num_layers, dropout=args.dropout).to(device)
    elif args.method == 'appnp':
        model = APPNP_Net(d, args.hidden_channels, c, alpha=args.gpr_alpha, dropout=args.dropout, num_layers=args.num_layers).to(device)
    elif args.method == 'gat':
        model = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                    dropout=args.dropout, heads=args.gat_heads, input_dropout= args.input_dropout).to(device)
 
    elif args.method == 'gat_pc':
        model = GAT_pc(d, args.hidden_channels, c, num_layers=args.num_layers,w1=args.w1,dropout=args.dropout, heads=args.gat_heads,
                       adj_order=args.adj_order, tebd_dim=args.tebd_dim, tdropout=args.tdropout, input_dropout= args.input_dropout, plus=args.plus).to(device)
    elif args.method == 'lp':
        mult_bin = args.dataset=='ogbn-proteins'
        model = MultiLP(c, args.lp_alpha, args.hops, mult_bin=mult_bin)
    elif args.method == 'mixhop':
        model = MixHop(d, args.hidden_channels, c, num_layers=args.num_layers,
                       dropout=args.dropout, hops=args.hops).to(device)
    elif args.method == 'gcnjk':
        model = GCNJK(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, jk_type=args.jk_type).to(device)
    elif args.method == 'gatjk':
        model = GATJK(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, heads=args.gat_heads,
                        jk_type=args.jk_type).to(device)
    elif args.method == 'h2gcn':
        model = H2GCN(d, args.hidden_channels, c, dataset.graph['edge_index'],
                        dataset.graph['num_nodes'],
                        num_layers=args.num_layers, dropout=args.dropout,
                        num_mlp_layers=args.num_mlp_layers).to(device)
  
    elif  args.method == 'h2gcn_pc':
        model = H2GCN_pc(d, args.hidden_channels, c, dataset.graph['edge_index'],
                         dataset.graph['num_nodes'],tebd_dim=args.tebd_dim,
                         num_layers=args.num_layers, dropout=args.dropout,
                         num_mlp_layers=args.num_mlp_layers, adj_order=args.adj_order, tdropout=args.tdropout, w1=args.w1, input_dropout= args.input_dropout, plus=args.plus).to(device)
    elif args.method == 'link_concat':
        model = LINK_Concat(d, args.hidden_channels, c, args.num_layers, dataset.graph['num_nodes'], dropout=args.dropout).to(device)
    elif args.method == 'linkx':
        model = LINKX(d, args.hidden_channels, c, args.num_layers, dataset.graph['num_nodes'],
        inner_activation=args.inner_activation, inner_dropout=args.inner_dropout, dropout=args.dropout, init_layers_A=args.link_init_layers_A, init_layers_X=args.link_init_layers_X).to(device)
   
    elif args.method == 'linkx_pc':
        model = LINKX_pc(d, args.hidden_channels, c, args.num_layers, dataset.graph['num_nodes'],
        inner_activation=args.inner_activation, inner_dropout=args.inner_dropout, 
        dropout=args.dropout, init_layers_A=args.link_init_layers_A, 
        init_layers_X=args.link_init_layers_X, tebd_dim=args.tebd_dim, adj_order=args.adj_order, w1 = args.w1, tdropout=args.tdropout, input_dropout = args.input_dropout, plus=args.plus
        ).to(device)
    elif args.method == 'gcn2':
        model = GCNII(d, args.hidden_channels, c, args.num_layers, args.gcn2_alpha, args.theta, dropout=args.dropout).to(device)
   
    elif args.method == 'gcn2_pc':
        model = GCNII_pc(d, args.hidden_channels, c, args.num_layers, args.gcn2_alpha, args.theta, dropout=args.dropout, adj_order=args.adj_order, tebd_dim=args.tebd_dim, tdropout=args.tdropout, w1=args.w1, input_dropout= args.input_dropout, plus=args.plus).to(device)
    else:
        raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):
    parser.add_argument('--dataset', type=str, default='fb100')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--method', '-m', type=str, default='link')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--hops', type=int, default=1,
                        help='power of adjacency matrix for certain methods')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--runs', type=int, default=10,
                        help='number of distinct runs')
    parser.add_argument('--eps', type=float, default=0.3, help='Fixed scalar or learnable weight.')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--gat_heads', type=int, default=8,
                        help='attention heads for gat')
    parser.add_argument('--lp_alpha', type=float, default=.1,
                        help='alpha for label prop')
    parser.add_argument('--gpr_alpha', type=float, default=.1,
                        help='alpha for gprgnn')
    parser.add_argument('--gcn2_alpha', type=float, default=.1,
                        help='alpha for gcn2')
    parser.add_argument('--theta', type=float, default=.5,
                        help='theta for gcn2')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    parser.add_argument('--jk_type', type=str, default='max', choices=['max', 'lstm', 'cat'],
                        help='jumping knowledge type')
    parser.add_argument('--rocauc', action='store_true',
                        help='set the eval function to rocauc')
    parser.add_argument('--num_mlp_layers', type=int, default=1,
                        help='number of mlp layers in h2gcn')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    # parser.add_argument('--re_split', type=str, default='11', help='example like: 0.1,0.1, present train_ratio and val_ratio')
    parser.add_argument('--adam', action='store_true', help='use adam instead of adamW')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--no_bn', action='store_true', help='do not use batchnorm')
    parser.add_argument('--sampling', action='store_true', help='use neighbor sampling')
    parser.add_argument('--inner_activation', action='store_true', help='Whether linkV3 uses inner activation')
    parser.add_argument('--inner_dropout', action='store_true', help='Whether linkV3 uses inner dropout')
    parser.add_argument("--SGD", action='store_true', help='Use SGD as optimizer')
    parser.add_argument('--link_init_layers_A', type=int, default=1)
    parser.add_argument('--link_init_layers_X', type=int, default=1)
    parser.add_argument('--stage', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.9)
    
    # there are various augmentation type
    # rfp: random edge flipping to -1 
    # sfp: static edge flipping to -1 
    parser.add_argument('--st_type', type=str, default='st', help='st, aug_st')
    parser.add_argument('--aug_type', type=str, default='rfp', help='random flip is fp')
    parser.add_argument('--fpp', type=float, default= 0.1, help='flip percent')
    parser.add_argument('--beta', type=float, default= 1., help='the strength of random flip update')
    
    # calibration 
    parser.add_argument('--smodel', type=str, default= 'gcn' )
    parser.add_argument('--cali', type=str, default= 'none' )
    parser.add_argument('--cali_epochs', type=int, default=200,
                    help='Number of epochs to calibration for self-training')
    parser.add_argument('--lr_for_cal', type=float, default=0.01)
    parser.add_argument('--l2_for_cal', type=float, default=5e-3,
                    help='Weight decay (L2 loss on parameters) for calibration.')
    parser.add_argument('--Lambda', type=float, default= 0., help='the coefficient of calibration loss')
    
    # transition loss 
    parser.add_argument('--transition_loss', type=str, default='none')
    parser.add_argument('--tt_coe', type=float, default=0.5)
    parser.add_argument('--dediag_coe', type=float, default=0.5)
    
    # focal loss 
    parser.add_argument('--focal_loss', type=str, default= 'none' )
    
    # gcnt 
    parser.add_argument('--w1', type=float, default= 0.05 )
    parser.add_argument('--adj_order', type=int, default= 2 )
    parser.add_argument('--tdropout', type=float, default= 0.1)
    parser.add_argument('--tebd_type', type=str, default= 'tebd')
    parser.add_argument('--tebd_dim', type=int, default= 128)
    
    # other params
    parser.add_argument('--re_split', action = 'store_true' )
    parser.add_argument('--paint_fig', action = 'store_true' )
    parser.add_argument('--paint_tsne', action = 'store_true' )
    parser.add_argument('--patience', type=int, default= 200 )
    parser.add_argument('--plus', action='store_false')
    parser.add_argument('--input_dropout', type=float, default=0.5)
    parser.add_argument('--treduc_save',action='store_true')