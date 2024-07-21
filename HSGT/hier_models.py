import math
import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Tuple, List
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_undirected
import torch_sparse
import pytorch_lightning as pl
import einops
from einops import rearrange, repeat
from graph import HierarchicalGraphDataBatch


full_query = False

class HistoricalEmbedding(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_nodes: int,
            device: str = 'cpu',
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        pin_memory = device == 'cpu'

        self.embedding = torch.empty(
            size=(num_nodes, hidden_dim),
            device=device,
            pin_memory=pin_memory
        )

    @torch.no_grad()
    def push(self, x: Tensor, node_indices: Tensor = None):
        if node_indices is None:
            assert x.shape == self.embedding.shape
            self.embedding = x
        else:
            self.embedding[node_indices] = x.to(self.embedding.device)

    @torch.no_grad()
    def pull(self, node_indices: Tensor = None) -> Tensor:
        if node_indices is None:
            return self.embedding
        else:
            return self.embedding[node_indices]

class StructuralEncoderLayer(nn.Module):
    def __init__(
            self,
            context_depth: int,
            random_connection_ratio: float,
    ):
        super().__init__()
        self.structural_embedding = nn.Embedding(context_depth + 1, embedding_dim=1, padding_idx=0)
        self.context_depth = context_depth
        self.random_connection_ratio = random_connection_ratio

    def forward(self, edge_index: Tensor, num_nodes: int):
        device = edge_index.device
        bias = (torch.rand((num_nodes, num_nodes), device=device) > self.random_connection_ratio).float()
        bias[bias == 1] = -torch.inf
        #bias = torch.zeros((num_nodes, num_nodes))
        #bias[:] = -torch.inf

        if edge_index.numel() == 0 or self.context_depth == 0:
            #bias.masked_fill_(bias == 0, -torch.inf)
            bias.fill_diagonal_(0)
            return bias

        dense_adj = to_dense_adj(edge_index=to_undirected(edge_index), max_num_nodes=num_nodes)[0]
        multihop_adjs = [dense_adj.clone(), torch.eye(num_nodes, device=device)]

        for _ in range(self.context_depth - 1):
            multihop_adjs.insert(0, multihop_adjs[0] @ dense_adj)

        for u, d in enumerate(multihop_adjs):
            bias[d > 0] = self.structural_embedding(torch.tensor(self.context_depth - u, device=device))

        bias = bias.to(edge_index.device)
        return bias


class TransformerBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            ffn_dim: int,
            num_heads: int,
            dropout_rates: List[float],
            batched: bool = False,
            use_simple_attn: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.h = num_heads
        self.d_k = hidden_dim // num_heads
        self.batched = batched
        self.use_simple_attn = use_simple_attn

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)

        if use_simple_attn:
            self.h_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.beta = torch.nn.Parameter(torch.ones(num_heads * self.d_k)*0.1)

        assert len(dropout_rates) == 4
        self.dropout = nn.ModuleList([nn.Dropout(p=dropout_rates[i]) for i in range(4)])
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.activation = F.gelu
    
    def attention(self, x_q: Tensor, x_kv: Tensor, bias: Optional[Tensor]) -> Tensor:
        if self.batched:
            q = rearrange(self.q_proj(x_q), "b q (h d) -> b h q d", h=self.h)
            k = rearrange(self.k_proj(x_kv), "b k (h d) -> b h k d", h=self.h)
            v = rearrange(self.v_proj(x_kv), "b k (h d) -> b h k d", h=self.h)
            scores = torch.einsum("b h q d, b h k d -> b h q k", q, k)
        else:
            q = rearrange(self.q_proj(x_q), "n (h d) -> h n d", h=self.h)
            k = rearrange(self.k_proj(x_kv), "n (h d) -> h n d", h=self.h)
            v = rearrange(self.v_proj(x_kv), "n (h d) -> h n d", h=self.h)
            scores = torch.einsum("h q d, h k d -> h q k", q, k)
        scores = scores / math.sqrt(self.d_k)
        if bias is not None:
            scores = scores + bias
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout[0](scores)
        if self.batched:
            x = torch.einsum("b h q k, b h k d -> b h q d", scores, v)
            x = rearrange(x, "b h q d -> b q (h d)").contiguous()
        else:
            x = torch.einsum("h q k, h k d -> h q d", scores, v)
            x = rearrange(x, "h q d -> q (h d)").contiguous()
        x = self.o_proj(x)

        # residual connection and FFN layers
        x = x_q + self.dropout[1](x)
        x2 = self.linear2(self.dropout[2](self.activation(self.linear1(self.norm1(x)))))
        x = x + self.dropout[3](x2)
        x = self.norm2(x)
        return x

    def simple_attention(self, x_q: Tensor, x_kv: Tensor) -> Tensor:
        d_k = self.d_k

        h = self.h_proj(x_q)
        q = F.sigmoid(self.q_proj(x_q)).view(-1, d_k, self.h)
        k = F.sigmoid(self.k_proj(x_q)).view(-1, d_k, self.h)
        v = self.v_proj(x_q).view(-1, d_k, self.h)

        # numerator
        kv = torch.einsum('ndh, nmh -> dmh', k, v)
        num = torch.einsum('ndh, dmh -> nmh', q, kv)

        # denominator
        k_sum = torch.einsum('ndh -> dh', k)
        den = torch.einsum('ndh, dh -> nh', q, k_sum).unsqueeze(1)

        # linear global attention based on kernel trick
        x = (num/den).reshape(-1, self.h * d_k)
        x = self.norm1(x) * (h + self.beta)
        x = F.relu(self.o_proj(x))
        x = self.dropout[-1](x)

        return x

    def forward(self, x_q: Tensor, x_kv: Tensor, bias: Optional[Tensor]) -> Tensor:
        if self.use_simple_attn:
            return self.simple_attention(x_q, x_kv)
        return self.attention(x_q, x_kv, bias)


class VerticalTransformerBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            ffn_dim: int,
            num_heads: int,
            dropout_rates: List[float],
    ):
        super().__init__()
        self.transformer = TransformerBlock(
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                dropout_rates=dropout_rates,
                use_simple_attn=False,
        )

    def forward(self, x_bottom: Tensor, x_upper: Tensor, hier_map: Tensor):
        num_bottom_nodes, num_upper_nodes = x_bottom.shape[0], x_upper.shape[0]
        bias = torch.zeros((num_upper_nodes, num_bottom_nodes), device=x_bottom.device)
        # aggregation attention mask. An upper node must only interact with bottom nodes it relates to.
        bias.masked_fill(F.one_hot(hier_map, num_upper_nodes).t() == 0, -torch.inf)
        x_upper = self.transformer(x_q=x_upper, x_kv=x_bottom, bias=bias)
        return x_upper


class HorizontalTransformerBlock(nn.Module):
    def __init__(
            self,
            num_blocks: int,
            hidden_dim: int,
            ffn_dim: int,
            num_heads: int,
            dropout_rates: List[float],
            context_depth: int,
            random_connection_ratio: float
    ):
        super().__init__()
        self.transformer = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                dropout_rates=dropout_rates,
                use_simple_attn=False,
            ) for _ in range(num_blocks)
        ])
        self.num_blocks = num_blocks
        self.structural_encoder = StructuralEncoderLayer(
            context_depth=context_depth,
            random_connection_ratio=random_connection_ratio
        )

    def forward(self, x: Tensor, edge_index: Tensor, num_target_nodes: int):
        bias = self.structural_encoder(
            edge_index=edge_index,
            num_nodes=x.shape[0]
        )
        for i in range(self.num_blocks):
            x = self.transformer[i](x_q=x, x_kv=x, bias=bias)
        return x


class ReadoutTransformerBlock(nn.Module):
    def __init__(
            self,
            num_blocks: int,
            hidden_dim: int,
            ffn_dim: int,
            num_heads: int,
            dropout_rates: List[float],
    ):
        super().__init__()
        self.transformer = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                dropout_rates=dropout_rates,
                batched=True
            ) for _ in range(num_blocks)
        ])
        self.num_blocks = num_blocks

    def forward(self, xs: List[Tensor]):
        xs = rearrange(xs, "l n d -> n l d")
        for i in range(self.num_blocks):
            xs = self.transformer[i](x_q=xs, x_kv=xs, bias=None)
        return xs[:, 0, :]


class HierarchicalGraphTransformer(pl.LightningModule):
    def __init__(
            self,
            # model args
            input_feature_dim: int,
            num_horizontal_blocks: int,
            num_readout_blocks: int,
            hidden_dim: int,
            ffn_dim: int,
            num_heads: int,
            num_hierarchies: int,
            transformer_dropout_rates: List[float],
            context_depth: int,
            random_connection_ratio: float,
            # data args
            out_channels: int,
            num_hierarchy_nodes: Optional[List[int]],
            # training args
            shared_params: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()

        # initiate model
        self.input_proj = nn.Linear(input_feature_dim, hidden_dim, bias=False)
        self.input_degree_encoder = nn.Embedding(200, hidden_dim)

        if shared_params:
            self.vertical_transformers = VerticalTransformerBlock(
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                dropout_rates=transformer_dropout_rates
            )
        else:
            self.vertical_transformers = nn.ModuleList([
                VerticalTransformerBlock(
                    hidden_dim=hidden_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    dropout_rates=transformer_dropout_rates
                ) for _ in range(num_hierarchies - 1)
            ])

        if shared_params:
            self.horizontal_transformers = HorizontalTransformerBlock(
                num_blocks=num_horizontal_blocks,
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                dropout_rates=transformer_dropout_rates,
                context_depth=context_depth,
                random_connection_ratio=random_connection_ratio
            )
        else:
            self.horizontal_transformers = nn.ModuleList([
                HorizontalTransformerBlock(
                    num_blocks=num_horizontal_blocks,
                    hidden_dim=hidden_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    dropout_rates=transformer_dropout_rates,
                    context_depth=context_depth,
                    random_connection_ratio=random_connection_ratio
                ) for _ in range(num_hierarchies)
            ])

        self.readout_transformers = ReadoutTransformerBlock(
            num_blocks=num_readout_blocks,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            dropout_rates=transformer_dropout_rates
        )

        if num_hierarchy_nodes is not None:
            self.histories = nn.ModuleList([
                HistoricalEmbedding(
                    hidden_dim=hidden_dim,
                    num_nodes=num_hierarchy_nodes[i]
                ) for i in range(num_hierarchies - 1)
            ])

        self.final_fc = nn.Linear(hidden_dim, out_channels)
        for p in self.parameters():
            if p.numel() > hidden_dim and p.dim() > 1:
                nn.init.xavier_normal_(p)

        self.num_hierarchies = num_hierarchies

    def forward(self, batched_data: HierarchicalGraphDataBatch):
        xs = [batched_data.data[i].x for i in range(batched_data.num_hierarchies)]
        target_nodes_hier_map = batched_data.hier_map
        num_target_nodes = batched_data.num_target_nodes
        node_indices = batched_data.node_indices

        # input projection
        xs[0] = self.input_proj(xs[0]) + self.input_degree_encoder(torch.clamp(batched_data.degrees[0], max=199)).int()
        for j in range(1, self.num_hierarchies):
            xs[j] = self.input_proj(xs[j])

        for i in range(self.num_hierarchies):
            if isinstance(self.horizontal_transformers, nn.ModuleList):
                xs[i] = self.horizontal_transformers[i](
                    x=xs[i], edge_index=batched_data.data[i].edge_index,
                    num_target_nodes=num_target_nodes[i]
                )
            else:
                xs[i] = self.horizontal_transformers(
                    x=xs[i], edge_index=batched_data.data[i].edge_index,
                    num_target_nodes=num_target_nodes[i]
                )
            if i < self.num_hierarchies - 1:
                x_bottom, x_upper = xs[i][:num_target_nodes[i]], xs[i + 1][:num_target_nodes[i + 1]]
                if isinstance(self.vertical_transformers, nn.ModuleList):
                    x_upper = self.vertical_transformers[i](
                        x_bottom=x_bottom, x_upper=x_upper,
                        hier_map=target_nodes_hier_map[i]
                    )
                else:
                    x_upper = self.vertical_transformers(
                        x_bottom=x_bottom, x_upper=x_upper,
                        hier_map=target_nodes_hier_map[i]
                    )

                # push and pull historical embeddings
                num_upper_nodes = num_target_nodes[i + 1]
                upper_node_indices = node_indices[i + 1][:num_upper_nodes]
                self.histories[i].push(x=x_upper, node_indices=upper_node_indices)

                neighbor_node_indices = node_indices[i + 1][num_upper_nodes:]  # indices for sampled neighborhood nodes
                x_neighbors = self.histories[i].pull(node_indices=neighbor_node_indices).to(x_upper.device)
                xs[i + 1] = torch.cat([x_upper, x_neighbors], dim=0)

        # retrive final representation
        target_node_hier_map = batched_data.hier_map
        cur_nodes = torch.arange(0, num_target_nodes[0])
        target_node_repr = [xs[0][cur_nodes]]
        for ind, mp in enumerate(target_node_hier_map):
            cur_nodes = mp[cur_nodes]
            target_node_repr.append(xs[ind + 1][cur_nodes])
        
        target_node_repr = self.readout_transformers(target_node_repr)
        output = self.final_fc(target_node_repr)
        output = F.log_softmax(output, dim=-1)
        return output
