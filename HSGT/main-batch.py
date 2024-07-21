import argparse
import sys
import time
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, subgraph, k_hop_subgraph
from torch_scatter import scatter

from tqdm import tqdm
from logger import Logger
from dataset import load_dataset
from data_utils import normalize, gen_normalized_adjs, eval_acc, eval_rocauc, eval_f1, \
    to_sparse_tensor, load_fixed_splits, adj_mul
from eval import evaluate_large, evaluate_batch
from parse import parse_method, parser_add_main_args
from graph import HierarchicalGraphData, HierarchyGenerator, HierarchyFeatureCollator, \
    HierarchicalClusterLoader


import warnings
warnings.filterwarnings('ignore')

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

# get the splits for all runs
if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [dataset.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                     for _ in range(args.runs)]
elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products', 'amazon2m', 'ogbn-papers100M', 'ogbn-papers100M-sub']:
    split_idx_lst = [dataset.load_fixed_splits()
                     for _ in range(args.runs)]
else:
    split_idx_lst = load_fixed_splits(n, args.data_dir, dataset, name=args.dataset, protocol=args.protocol, runs=args.runs)

# whether or not to symmetrize
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

### Graph coasening ###
num_hierarchy_nodes = [int(n * 0.05)]
print("number of hierarchy nodes: ", num_hierarchy_nodes)
num_sampled_neighbors = [[5, 10], [5]]
num_hierarchies = len(num_hierarchy_nodes) + 1
args.num_hierarchies = num_hierarchies
args.num_hierarchy_nodes = num_hierarchy_nodes

hierarchy_generator = HierarchyGenerator(
    generator_type="metis",
    num_nodes=num_hierarchy_nodes,
    dataset_name=args.dataset
)

feature_collator = HierarchyFeatureCollator("mean")
input_dataset = dataset.data
if isinstance(input_dataset, torch_geometric.data.Dataset):
    input_dataset = input_dataset[0]

hierarchical_dataset = HierarchicalGraphData(
    graph_data=input_dataset,
    hierarchy_generator=hierarchy_generator,
    feature_collator=feature_collator,
    num_target_nodes=n
)

### Load method ###
model = parse_method(args, c, d, device)

### Loss function (Single-class, Multi-class) ###
if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC, F1) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']

if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
    if dataset.label.shape[1] == 1:
        true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
    else:
        true_label = dataset.label
else:
    true_label = dataset.label

### Training loop ###
for run in range(args.runs):
    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_hier_loader = HierarchicalClusterLoader(
        data=hierarchical_dataset,
        num_neighbors=num_sampled_neighbors,
        batch_size=args.batch_size,
        num_workers=10,
        shuffle=True,
        input_nodes=split_idx['train']
    )
    eval_hier_loader = HierarchicalClusterLoader(
        data=hierarchical_dataset,
        num_neighbors=num_sampled_neighbors,
        batch_size=args.batch_size,
        num_workers=10,
        shuffle=False,
        input_nodes=split_idx["valid"],
    )
    test_hier_loader = HierarchicalClusterLoader(
        data=hierarchical_dataset,
        num_neighbors=num_sampled_neighbors,
        batch_size=args.batch_size,
        num_workers=10,
        shuffle=False,
        input_nodes=split_idx["test"],
    )

    # model.reset_parameters()
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')

    for epoch in range(args.epochs):
        model.to(device)
        model.train()

        start = time.time()
        for batched_data in train_hier_loader:
            if not batched_data.is_valid_batch:
                continue
            batched_data.to(device)
            y_i = batched_data.y
            optimizer.zero_grad()
            out_i = model(batched_data)[batched_data.supervised_node_indices]
            if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
                loss = criterion(out_i, y_i)
            else:
                out_i = F.log_softmax(out_i, dim=1)
                loss = criterion(out_i, y_i)
            loss.backward()
            optimizer.step()
        epoch_time = time.time() - start

        if epoch % args.eval_step == 0:
            result = evaluate_batch(model, [train_hier_loader, eval_hier_loader, test_hier_loader], device)
            logger.add_result(run, result[:-1])

            if epoch % args.display_step == 0:
                print_str = f'Epoch: {epoch:02d}, ' + \
                            f'Loss: {loss:.4f}, ' + \
                            f'Train: {100 * result[0]:.2f}%, ' + \
                            f'Valid: {100 * result[1]:.2f}%, ' + \
                            f'Test: {100 * result[2]:.2f}%, ' + \
                            f'Time: {epoch_time:.2f}s'
                print(print_str)
    logger.print_statistics(run)

logger.print_statistics()