import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.utils import subgraph

@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out

@torch.no_grad()
def evaluate_large(model, dataset, split_idx, eval_func, criterion, args, device="cpu", result=None):
    if result is not None:
        out = result
    else:
        model.eval()

    model.to(torch.device(device))
    dataset.label = dataset.label.to(torch.device(device))
    edge_index, x = dataset.graph['edge_index'].to(torch.device(device)), dataset.graph['node_feat'].to(torch.device(device))
    out = model(x, edge_index)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out

def eval_acc_batch(model, data_loader, device):
    total, correct = 0, 0
    for batched_data in data_loader:
        if not batched_data.is_valid_batch:
            continue
        batched_data.to(device)
        y_i = batched_data.y
        out_i = model(batched_data)[batched_data.supervised_node_indices]

        cur_valid_total, cur_valid_correct = eval_acc(y_i, out_i)
        total += cur_valid_total
        correct += cur_valid_correct

    acc = correct / total
    return acc


def evaluate_batch(model, data_loaders, device):
    model.to(device)
    model.eval()
    train_dataloader, eval_dataloader, test_dataloader = data_loaders

    with torch.no_grad():
        train_acc = eval_acc_batch(model, train_dataloader, device)
        valid_acc = eval_acc_batch(model, eval_dataloader, device)
        test_acc = eval_acc_batch(model, test_dataloader, device)

    return train_acc, valid_acc, test_acc, 0, None


def eval_acc(true, pred):
    '''
    true: (n, 1)
    pred: (n, c)
    '''
    pred=torch.max(pred, dim=1)[1]
    true_cnt=(true==pred).sum()

    return true.shape[0], true_cnt.item()


if __name__=='__main__':
    x=torch.arange(4).unsqueeze(1)
    y=torch.Tensor([[3,0,0,0],
                    [3,2,1.5,2.8],
                    [0,0,2,1],
                    [0,0,1,3]
                    ])
    a, b=eval_acc(x, y)
    print(x)
    print(a,b)
