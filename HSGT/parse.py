from gnns import *
from hier_models import *

def parse_method(args, c, d, device):
    if args.method == 'gcn':
        model = GCN(in_channels=d,
                    hidden_channels=args.hidden_channels,
                    out_channels=c,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    use_bn=args.use_bn).to(device)
    elif args.method == 'gat':
        model = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                    dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads).to(device)
    elif args.method == 'mlp' or args.method == 'manireg':
        model = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                    out_channels=c, num_layers=args.num_layers,
                    dropout=args.dropout).to(device)
    elif args.method == 'heat':
        model = GCN(in_channels=d,
                    hidden_channels=args.hidden_channels,
                    out_channels=c,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    use_bn=args.use_bn).to(device)
    elif args.method == 'sgc':
        # model = SGC(in_channels=d, out_channels=c, hops=args.hops).to(device)
        model = SGCMem(in_channels=d, out_channels=c, hops=args.hops, use_bn=args.use_bn).to(device)
    elif args.method=='sgc2':
        model=SGC2(d,args.hidden_channels,c,args.hops,args.num_layers,args.dropout, use_bn=args.use_bn).to(device)
    elif args.method=='sign':
        model=SIGN(in_channels=d,hidden_channels=args.hidden_channels,
                    out_channels=c, hops=args.hops, num_layers=args.num_layers,
                    dropout=args.dropout,use_bn=args.use_bn).to(device)
    elif args.method == 'hsgt':
        model = HierarchicalGraphTransformer(d, args.num_horizontal_layer, 1, args.hidden_channels, args.hidden_channels, 
                                             args.trans_num_heads, args.num_hierarchies, 
                                             transformer_dropout_rates=[args.dropout for _ in range(4)],
                                             context_depth=2,
                                             random_connection_ratio=0.1,
                                             out_channels=c,
                                             num_hierarchy_nodes=args.num_hierarchy_nodes,
                                             shared_params=args.shared_params,
                                             ).to(device)
    else:
        raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--protocol', type=str, default='semi',
                        help='protocol for cora datasets, semi or supervised')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--label_num_per_class', type=int, default=20, help='labeled nodes randomly selected')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'],
                        help='evaluation metric')

    # gnn branch
    parser.add_argument('--method', type=str, default='sgformer')
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--use_graph', action='store_true', help='use input graph')
    parser.add_argument('--aggregate', type=str, default='add', help='aggregate type, add or cat.')
    parser.add_argument('--graph_weight', type=float, default=0.8, help='graph weight.')

    # all-pair attention (Transformer) branch
    parser.add_argument('--trans_num_heads', type=int, default=1, help='number of heads for attention')
    parser.add_argument('--trans_use_weight', action='store_true', help='use weight for trans convolution')
    parser.add_argument('--trans_use_bn', action='store_true', help='use layernorm for trans')
    parser.add_argument('--trans_use_residual', action='store_true', help='use residual link for each trans layer')
    parser.add_argument('--trans_use_act', action='store_true', help='use activation for each trans layer')
    parser.add_argument('--trans_num_layers', type=int, default=2, help='number of layers for all-pair attention.')
    parser.add_argument('--trans_weight_decay', type=float, default=1e-3)
    parser.add_argument('--shared_params', type=bool, default=False)

    # hier transformer
    parser.add_argument('--num_horizontal_layer', type=int, default=3, help='number of horizontal layers')
    parser.add_argument('--num_readout_layer', type=int, default=1, help='number of readout layers')

    # training
    parser.add_argument('--dropout', type=float, help='dropout rate', default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=10000, help='mini batch training for large graphs')
    parser.add_argument('--patience', type=int, default=200, help='early stopping patience.')

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--eval_step', type=int,
                        default=1, help='how often to evaluate')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')
    parser.add_argument('--save_result', action='store_true',
                        help='save result')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--use_pretrained', action='store_true', help='whether to use pretrained model')
    parser.add_argument('--save_att', action='store_true', help='whether to save attention (for visualization)')
    parser.add_argument('--model_dir', type=str, default='../../model/')

    # other gnn parameters (for baselines)
    parser.add_argument('--hops', type=int, default=2,
                        help='number of hops for SGC')
    parser.add_argument('--gat_heads', type=int, default=4,
                        help='attention heads for gat')
    parser.add_argument('--out_heads', type=int, default=1,
                        help='out heads for gat')


