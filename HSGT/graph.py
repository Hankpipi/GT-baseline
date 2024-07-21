import os
import torch
from torch import Tensor
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import degree
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.nn.pool.pool import pool_edge
from torch_scatter import scatter_add, scatter
from torch.utils.data import DataLoader
from torch_geometric.sampler.utils import to_csc
from torch_geometric.loader.utils import filter_data
import torch_cluster
import torch_geometric.transforms as T
from typing import List, Tuple, Optional
import copy
import time


def rearrange_cluster_map(cluster_map: Tensor) -> Tensor:
    vals = cluster_map.unique().sort()[0].tolist()
    vals_map = {val: i for i, val in enumerate(vals)}
    # cluster_map_list = list(cluster_map)
    rearranged_map = list(map(lambda x: vals_map[int(x)], cluster_map))
    return torch.tensor(rearranged_map)


class HierarchyGenerator:
    def __init__(
            self,
            generator_type: str,
            num_nodes: List[int],
            dataset_name: Optional[str] = None
    ):
        self.generator_type = generator_type
        self.dataset_name = dataset_name
        if generator_type == "random":
            self.coarsener = RandomHierarchyGenerator(num_nodes)
        elif generator_type == "metis":
            self.coarsener = MetisHierarchyGenerator(num_nodes)
        else:
            raise NotImplementedError

    def __call__(self, data: torch_geometric.data.Data):
        return self.coarsener(data, self.dataset_name)


class RandomHierarchyGenerator:
    def __init__(self, num_nodes: List[int]):
        self.num_layers = len(num_nodes)
        self.num_nodes = num_nodes
        print(f"Random hierarchy generator defined, {self.num_layers} layers in total,\
              nodes in every layer: {num_nodes}.")

    def __call__(self, data: torch_geometric.data.Data, dataset_name: Optional[str]) -> List[Tensor]:
        hier_map = []
        sizes = [data.num_nodes]
        for i in range(self.num_layers):
            cluster_map = rearrange_cluster_map(torch.randint(0, self.num_nodes[i], (sizes[i],)))
            hier_map.append(cluster_map)
            sizes.append(cluster_map.max() + 1)
        return hier_map


class MetisHierarchyGenerator:
    def __init__(self, num_nodes: List[int]):
        self.num_layers = len(num_nodes)
        self.num_nodes = num_nodes

    def __call__(self, data: torch_geometric.data.Data, dataset_name: Optional[str]) -> List[Tensor]:
        save_path = None
        if dataset_name is not None:
            save_path = f"./saved_partition/metis_{dataset_name}_{self.num_nodes}.pt"
            if os.path.exists(save_path):
                hier_map = torch.load(save_path)
                print(f"Loaded saved partition file {save_path}.")
                return hier_map

        edge_index = data.edge_index
        transform = T.to_sparse_tensor.ToSparseTensor()

        adj_t = transform(copy.deepcopy(data)).adj_t
        adj_t = adj_t.to_symmetric()
        part_fn = torch.ops.torch_sparse.partition
        counter = time.perf_counter()
        hier_map = []
        print(f"Starting Metis partitioning...", flush=True)

        for i in range(self.num_layers):
            if self.num_nodes[i] == 1:
                hier_map.append(torch.zeros(self.num_nodes[i - 1], dtype=int))
                continue
            row_ptr, col, _ = adj_t.csr()
            cluster_map = part_fn(row_ptr, col, None, self.num_nodes[i], False)
            hier_map.append(cluster_map)
            edge_index, _ = pool_edge(cluster=cluster_map, edge_index=edge_index)
            next_data = transform(torch_geometric.data.Data(edge_index=edge_index, num_nodes=self.num_nodes[i]))
            adj_t = next_data.adj_t
            adj_t = adj_t.to_symmetric()

        print(f"Done, time = {time.perf_counter() - counter:.2f}s.", flush=True)
        #print(f"hierarchy map: {[t.shape for t in hier_map]}", flush=True)
        if save_path is not None:
            torch.save(hier_map, save_path)

        return hier_map


class HierarchyFeatureCollator:
    def __init__(
            self,
            collator_type: str
    ):
        self.collator_type = collator_type

    def __call__(
            self,
            data: torch_geometric.data.Data,
            hier_map: List[Tensor]
    ) -> List[Tensor]:
        collated_feature = [data.x]
        if self.collator_type == "mean":
            for i in range(len(hier_map)):
                collated_feature.append(scatter(
                    src=collated_feature[i],
                    index=hier_map[i],
                    dim=0,
                    reduce="mean"
                ))
        else:
            raise NotImplementedError("Feature Collator undefined.")
        return collated_feature


class HierarchyFeatureCollator:
    def __init__(
            self,
            collator_type: str
    ):
        self.collator_type = collator_type

    def __call__(
            self,
            data: torch_geometric.data.Data,
            hier_map: List[Tensor]
    ) -> List[Tensor]:
        collated_feature = [data.x]
        if self.collator_type == "mean":
            for i in range(len(hier_map)):
                collated_feature.append(scatter(
                    src=collated_feature[i],
                    index=hier_map[i],
                    dim=0,
                    reduce="mean"
                ))
        else:
            raise NotImplementedError("Feature Collator undefined.")
        return collated_feature


class HierarchicalGraphData:
    def __init__(
            self,
            graph_data: torch_geometric.data.Data,
            hierarchy_generator: HierarchyGenerator,
            feature_collator: HierarchyFeatureCollator,
            num_target_nodes: int,
    ):
        self.org_data = graph_data
        self.y = graph_data.y
        self.num_leaf_nodes: int = graph_data.num_nodes
        self.hier_map = hierarchy_generator(graph_data)  # List[Tensor]
        self.num_hierarchies = len(self.hier_map) + 1
        self.collated_feature = feature_collator(graph_data, self.hier_map)
        self.num_target_nodes = num_target_nodes

        self.hier_data = [self.org_data]
        self.degrees = [degree(
            index=self.org_data.edge_index[0, :],
            num_nodes=self.num_leaf_nodes
        ).int()]
        coarsened_edge_index = self.org_data.edge_index

        for i in range(len(self.hier_map)):
            num_nodes = self.collated_feature[i + 1].shape[0]
            coarsened_edge_index, _ = pool_edge(
                cluster=self.hier_map[i],
                edge_index=coarsened_edge_index,
            )
            coarsened_data = Data(
                x=self.collated_feature[i + 1],
                edge_index=coarsened_edge_index,
                num_nodes=num_nodes
            )
            self.hier_data.append(coarsened_data)
            # calculate node degrees
            coarsened_data_degree = degree(
                index=coarsened_data.edge_index[0, :],
                num_nodes=num_nodes
            ).int()
            if coarsened_data_degree.numel() == 0:
                coarsened_data_degree = torch.tensor([0], dtype=torch.int)
            self.degrees.append(coarsened_data_degree)
            print(self.degrees)

    def to(self, device):
        for i in range(len(self.hier_data)):
            self.hier_data[i] = self.hier_data[i].to(device)
            self.degrees[i] = self.degrees[i].to(device)
        return self

    def __len__(self):
        return self.num_hierarchies

    def __getitem__(self, ind: int):
        return self.hier_data[ind]


class HierarchicalGraphDataBatch:
    def __init__(
            self,
            is_valid_batch: bool,
            data: Optional[List[Data]] = None,
            node_indices: Optional[List[Tensor]] = None,
            hier_map: Optional[List[Tensor]] = None,
            num_target_nodes: Optional[List[int]] = None,
            degrees: Optional[List[Tensor]] = None
    ):
        self.is_valid_batch = is_valid_batch
        if is_valid_batch:
            self.data = data
            self.node_indices = node_indices
            self.hier_map = hier_map
            self.num_target_nodes = num_target_nodes
            self.num_hierarchies = len(self.data)
            self.degrees = degrees
            self.y = None
            self.supervised_node_indices = None

    def __getitem__(self, ind: int):
        return self.data[ind], self.hier_map[ind], self.num_target_nodes[ind]

    def to(self, device):
        if self.is_valid_batch:
            self.data = [d.to(device) for d in self.data]
            # self.node_indices = [d.to(device) for d in self.node_indices]
            self.hier_map = [d.to(device) for d in self.hier_map]
            self.degrees = [d.to(device) for d in self.degrees]
            # self.target_nodes_edge_index = [d.to(device) for d in self.target_nodes_edge_index]
            if self.y is not None:
                self.y = self.y.to(device)
            if self.supervised_node_indices is not None:
                self.supervised_node_indices = self.supervised_node_indices.to(device)


class HierarchicalDataNeighborSampler:
    def __init__(
            self,
            data: HierarchicalGraphData,
            num_neighbors: List[List[int]],
            replace: bool = False,
            directed: bool = False,
    ):
        self.data = data
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.directed = directed
        self.num_hierarchies = data.num_hierarchies
        self.hier_map = data.hier_map
        self.csc_tensors = []
        for i in range(self.num_hierarchies):
            self.csc_tensors.append(to_csc(self.data[i]))  # (colptr, row, perm)

    def __call__(self, target_nodes: List[List[int]]):
        sampled_data, sampled_node_degrees, sampled_node_indices = [], [], []
        sample_fn = torch.ops.torch_sparse.neighbor_sample

        for i in range(self.num_hierarchies):
            data_colptr, data_row, data_perm = self.csc_tensors[i]
            node, row, col, edge = sample_fn(
                data_colptr,
                data_row,
                torch.tensor(target_nodes[i]),
                self.num_neighbors[i],
                self.replace,
                self.directed,
            )
            data = filter_data(self.data[i], node, row, col, edge, data_perm)
            sampled_node_indices.append(node.clone())
            sampled_data.append(data)
            sampled_node_degrees.append(self.data.degrees[i][node])

        return HierarchicalGraphDataBatch(
            is_valid_batch=True,
            data=sampled_data,
            node_indices=sampled_node_indices,
            hier_map=None,
            num_target_nodes=None,
            degrees=sampled_node_degrees,
        )


class HierarchicalClusterLoader(DataLoader):
    def __init__(
            self,
            data: HierarchicalGraphData,
            num_neighbors: List[List[int]],
            batch_size: int,
            num_workers: int,
            shuffle: bool,
            input_nodes: Tensor,
    ):
        self.data = data
        self.num_top_clusters = self.data[-1].num_nodes
        self.num_hierarchies = data.num_hierarchies

        self.inverse_hier_map = []
        for i in range(self.num_hierarchies - 1):
            inverse_map = {j: [] for j in range(self.data[i + 1].num_nodes)}
            for u, v in enumerate(self.data.hier_map[i]):
                inverse_map[int(v)].append(u)
            self.inverse_hier_map.append(inverse_map)

        self.neighbor_sampler = HierarchicalDataNeighborSampler(
            data=data, num_neighbors=num_neighbors,
            replace=False, directed=False
        )
        self.input_nodes = input_nodes
        self.input_nodes_set = set(input_nodes.tolist())

        if batch_size == -1:
            batch_size = self.num_top_clusters

        super().__init__(
            dataset=torch.arange(0, self.num_top_clusters),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate,
            pin_memory=True
        )

    def collate(self, cluster_index):
        # parameter cluster_index index top-level clusters
        target_nodes = [[int(l) for l in cluster_index]]
        target_nodes_hier_map = []
        for i in range(self.num_hierarchies - 2, -1, -1):
            target_nodes.insert(0, [])
            target_nodes_hier_map.insert(0, [])
            for u, j in enumerate(target_nodes[1]):
                target_nodes[0] += self.inverse_hier_map[i][j]
                target_nodes_hier_map[0] += [u for _ in range(len(self.inverse_hier_map[i][j]))]

        target_nodes_hier_map = [torch.tensor(m) for m in target_nodes_hier_map]

        supervised_nodes = []
        for u, v in enumerate(target_nodes[0]):
            if v in self.input_nodes_set:
                supervised_nodes.append(u)
        if len(supervised_nodes) == 0:
            return HierarchicalGraphDataBatch(is_valid_batch=False)

        supervised_nodes = torch.tensor(supervised_nodes)

        y = self.data.y[target_nodes[0]]
        if y.dim() == 2 and y.shape[1] == 1:
            y = y[:, 0]

        num_target_nodes = [len(d) for d in target_nodes]
        sampled_data = self.neighbor_sampler(target_nodes)
        sampled_data.hier_map = target_nodes_hier_map
        sampled_data.num_target_nodes = num_target_nodes
        sampled_data.y = y[supervised_nodes]
        sampled_data.supervised_node_indices = supervised_nodes
        return sampled_data