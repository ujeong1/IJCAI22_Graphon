from time import time
import logging
import os
import os.path as osp
import numpy as np
import time

from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from torch.autograd import Variable

import random
from torch.optim.lr_scheduler import StepLR

from utils import stat_graph, split_class_graphs, align_graphs
from utils import two_graphons_mixup, universal_svd
from graphon_estimator import universal_svd
from models import GIN, GCN

import argparse

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d')


def prepare_dataset_x(dataset):
    if dataset[0].x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
            data.num_nodes = int(torch.max(data.edge_index)) + 1

        if max_degree < 2000:
            # dataset.transform = T.OneHotDegree(max_degree)

            for data in dataset:
                degs = degree(data.edge_index[0], dtype=torch.long)
                data.x = F.one_hot(degs, num_classes=max_degree + 1).to(torch.float)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            for data in dataset:
                degs = degree(data.edge_index[0], dtype=torch.long)
                data.x = ((degs - mean) / std).view(-1, 1)
    return dataset


def prepare_dataset_onehot_y(dataset):
    y_set = set()
    for data in dataset:
        y_set.add(int(data.y))
    num_classes = len(y_set)

    for data in dataset:
        data.y = F.one_hot(data.y, num_classes=num_classes).to(torch.float)[0]
    return dataset


def mixup_cross_entropy_loss(input, target, size_average=True):
    """Origin: https://github.com/moskomule/mixup.pytorch
    in PyTorch's cross entropy, targets are expected to be labels
    so to predict probabilities this loss is needed
    suppose q is the target and p is the input
    loss(p, q) = -\sum_i q_i \log p_i
    """
    assert input.size() == target.size()
    assert isinstance(input, Variable) and isinstance(target, Variable)
    loss = - torch.sum(input * target)
    return loss / input.size()[0] if size_average else loss


def train(model, train_loader):
    model.train()
    loss_all = 0
    graph_all = 0
    for data in train_loader:
        # print( "data.y", data.y )
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        y = data.y.view(-1, num_classes)
        loss = mixup_cross_entropy_loss(output, y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        graph_all += data.num_graphs
        optimizer.step()
    loss = loss_all / graph_all
    return model, loss


def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        y = data.y.view(-1, num_classes)
        loss += mixup_cross_entropy_loss(output, y).item() * data.num_graphs
        y = y.max(dim=1)[1]
        correct += pred.eq(y).sum().item()
        total += data.num_graphs
    acc = correct / total
    loss = loss / total
    return acc, loss


def sample_from_graphon(graphons, num_augs):
    pyg_graphs = []
    for label, graphon in graphons:
        sample_graph_label = torch.from_numpy(label).type(torch.float32)
        for i in range(num_augs):
            sample_graph = (np.random.rand(*graphon.shape) <= graphon).astype(np.int32)
            sample_graph = np.triu(sample_graph)
            sample_graph = sample_graph + sample_graph.T - np.diag(np.diag(sample_graph))

            sample_graph = sample_graph[sample_graph.sum(axis=1) != 0]
            sample_graph = sample_graph[:, sample_graph.sum(axis=0) != 0]

            A = torch.from_numpy(sample_graph)
            edge_index, _ = dense_to_sparse(A)
            num_nodes = int(torch.max(edge_index)) + 1

            pyg_graph = Data()
            pyg_graph.y = sample_graph_label
            pyg_graph.num_nodes = num_nodes
            pyg_graph.edge_index = edge_index
            pyg_graphs.append(pyg_graph)
    return pyg_graphs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./")
    parser.add_argument('--dataset', type=str, default="REDDIT-BINARY")
    parser.add_argument('--model', type=str, default="GIN")
    parser.add_argument('--epoch', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--gmixup', type=str, default="False")
    parser.add_argument('--lam_range', type=str, default="[0.005, 0.01]")
    parser.add_argument('--aug_ratio', type=float, default=0.15)
    parser.add_argument('--aug_num', type=int, default=10)
    parser.add_argument('--gnn', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=1314)
    parser.add_argument('--log_screen', type=str, default="False")
    parser.add_argument('--ge', type=str, default="MC")

    args = parser.parse_args()
    print(args)
    data_path = args.data_path
    dataset_name = args.dataset
    seed = args.seed
    lam_range = eval(args.lam_range)
    log_screen = eval(args.log_screen)
    gmixup = eval(args.gmixup)
    num_epochs = args.epoch

    num_hidden = args.num_hidden
    batch_size = args.batch_size
    learning_rate = args.lr
    ge = args.ge
    aug_ratio = args.aug_ratio
    aug_num = args.aug_num
    model = args.model

    if log_screen is True:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.info('parser.prog: {}'.format(parser.prog))
    logger.info("args:{}".format(args))

    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"runing device: {device}")

    path = osp.join(data_path, dataset_name)
    dataset = TUDataset(path, name=dataset_name)
    dataset = list(dataset)

    for graph in dataset:
        graph.y = graph.y.view(-1)

    dataset = prepare_dataset_onehot_y(dataset)

    random.seed(seed)
    random.shuffle(dataset)


    def K_shot_select(train_dataset, K):
        classes = [str(data.y) for data in train_dataset]
        classes = list(set(classes))
        class_idx = dict()
        for label in classes:
            class_idx[label] = []
        for i, data in enumerate(train_dataset):
            label = str(data.y)
            if len(class_idx[label]) >= K:
                continue
            class_idx[label].append(i)
        result = []
        for label, idx_list in class_idx.items():
            result += [train_dataset[idx] for idx in idx_list]
        return result


    train_nums = int(len(dataset) * 0.7)
    train_dataset = dataset[:train_nums]
    random.shuffle(train_dataset)
    K = 10
    train_dataset = K_shot_select(train_dataset, K)
    train_val_nums = int(len(dataset) * 0.8)
    val_nums = train_val_nums - train_nums

    avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(
        dataset[: train_nums])
    logger.info(f"avg num nodes of training graphs: {avg_num_nodes}")
    logger.info(f"avg num edges of training graphs: {avg_num_edges}")
    logger.info(f"avg density of training graphs: {avg_density}")
    logger.info(f"median num nodes of training graphs: {median_num_nodes}")
    logger.info(f"median num edges of training graphs: {median_num_edges}")
    logger.info(f"median density of training graphs: {median_density}")

    resolution = int(median_num_nodes)

    # if gmixup == True:
    class_graphs = split_class_graphs(train_dataset)
    graphons = []
    # for label, graphs in class_graphs:
    #
    #         logger.info(f"label: {label}, num_graphs:{len(graphs)}" )
    #         align_graphs_list, normalized_node_degrees, max_num, min_num = align_graphs(
    #             graphs, padding=True, N=resolution)
    #         logger.info(f"aligned graph {align_graphs_list[0].shape}" )
    #
    #         logger.info(f"ge: {ge}")
    for label, labeled_graphs in class_graphs:
        align_graphs_list, normalized_node_degrees, max_num, min_num = align_graphs(
            labeled_graphs, padding=True, N=resolution)
        graphon = universal_svd(align_graphs_list, threshold=0.2)
        graphons.append((label, graphon))

    for label, graphon in graphons:
        logger.info(f"graphon info: label:{label}; mean: {graphon.mean()}, shape, {graphon.shape}")
    num_classes = len(class_graphs)
    num_augs = 100  # int(train_nums * aug_ratio / aug_num)
    # num_augs = int((len(dataset)*0.2)/num_classes)
    # lam_list = np.random.uniform(low=lam_range[0], high=lam_range[1], size=(aug_num,))

    random.seed(seed)
    new_graph = []
    # for lam in lam_list:
    #     logger.info(f"lam: {lam}")
    #     logger.info(f"num_sample: {num_sample}")
    #     two_graphons = random.sample(graphons, 2)
    #     new_graph += two_graphons_mixup(two_graphons, la=lam, num_sample=num_sample)
    #     logger.info(f"label: {new_graph[-1].y}")
    new_graph = sample_from_graphon(graphons, num_augs)

    avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(
        new_graph)
    logger.info(f"avg num nodes of new graphs: {avg_num_nodes}")
    logger.info(f"avg num edges of new graphs: {avg_num_edges}")
    logger.info(f"avg density of new graphs: {avg_density}")
    logger.info(f"median num nodes of new graphs: {median_num_nodes}")
    logger.info(f"median num edges of new graphs: {median_num_edges}")
    logger.info(f"median density of new graphs: {median_density}")
    del dataset[:train_nums]
    train_dataset = train_dataset + new_graph
    dataset = train_dataset + dataset
    train_nums = len(train_dataset)
    train_val_nums = train_nums + val_nums

    print("*" * 100)
    print("synthetic graphs:", len(new_graph))
    # print("training graphs:", len(train_dataset))
    print("final train num", train_nums)
    print("final val num", train_val_nums-train_nums)
    print("final test num", len(dataset) - train_val_nums)
    print("*" * 100)
    # logger.info(f"real aug ratio: {len(new_graph) / train_nums}")

    dataset = prepare_dataset_x(dataset)

    logger.info(f"num_features: {dataset[0].x.shape}")
    logger.info(f"num_classes: {dataset[0].y.shape}")

    num_features = dataset[0].x.shape[1]
    num_classes = dataset[0].y.shape[0]
    train_dataset = dataset[:train_nums]
    val_dataset = dataset[train_nums:train_val_nums]
    test_dataset = dataset[train_val_nums:]

    logger.info(f"train_dataset size: {len(train_dataset)}")
    logger.info(f"val_dataset size: {len(val_dataset)}")
    logger.info(f"test_dataset size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    if model == "GIN":
        model = GIN(num_features=num_features, num_classes=num_classes, num_hidden=num_hidden).to(device)
    elif model == "GCN":
        model = GCN(num_features=num_features, num_classes=num_classes, num_hidden=num_hidden).to(device)
    else:
        logger.info(f"No model.")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
    best_val_acc = 0

    for epoch in range(1, num_epochs):
        model, train_loss = train(model, train_loader)
        train_acc = 0
        val_acc, val_loss = test(model, val_loader)
        scheduler.step()

        logger.info(
            'Epoch: {:03d}, Train Loss: {:.6f}, Val Loss: {:.6f}, Val Acc: {: .6f}'.format(
                epoch, train_loss, val_loss, val_acc))
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            import copy
            state_dict = copy.deepcopy(model.state_dict())
    model.load_state_dict(state_dict)
    test_acc, test_loss = test(model, test_loader)

    with open("result_aug.txt", "a") as f:
        f.write(str(test_acc) + "\n")
