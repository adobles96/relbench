from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP, PositionalEncoding, HeteroConv, GATConv, LayerNorm
from torch_geometric.typing import NodeType, EdgeType

from relbench.external.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder


class HeteroTemporalEdgeEncoder(torch.nn.Module):
    def __init__(self, num_edge_features: Dict[EdgeType, int]):
        super().__init__()

        # TODO modify to use fixed channel size once you implement edge feature encoder
        self.encoder_dict = torch.nn.ModuleDict({
            edge_type: PositionalEncoding(n_feats)
            for edge_type, n_feats in num_edge_features.items()
        })
        self.lin_dict = torch.nn.ModuleDict({
            edge_type: torch.nn.Linear(n_feats, n_feats)
            for edge_type, n_feats in num_edge_features.items()
        })

    def reset_parameters(self):
        for encoder in self.encoder_dict.values():
            encoder.reset_parameters()
        for lin in self.lin_dict.values():
            lin.reset_parameters()

    def forward(
        self,
        seed_time: Tensor,
        time_dict: Dict[EdgeType, Tensor],
        batch_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor]
    ) -> Dict[NodeType, Tensor]:
        out_dict: Dict[NodeType, Tensor] = {}

        for edge_type, time in time_dict.items():
            node_type_src = edge_type[0]
            # use source node indices to get seed time
            src_node_indices = edge_index_dict[edge_type][0]

            rel_time = seed_time[batch_dict[node_type_src][src_node_indices]] - time
            rel_time = rel_time / (60 * 60 * 24)  # Convert seconds to days.

            x = self.encoder_dict[edge_type](rel_time)
            x = self.lin_dict[edge_type](x)
            out_dict[edge_type] = x

        return out_dict


# potentially useful: https://github.com/pyg-team/pytorch_geometric/discussions/6869
class HeteroGraphGAT(torch.nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        num_edge_features: Dict[str, int],
        channels: int,
        heads: int = 1,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        assert channels % heads == 0, "channels should be divisible by heads"
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    # NOTE: output has size (num_nodes, channels * heads) so if heads > 1
                    # then either set out_channels channels/heads or use MLP to downscale channels
                    edge_type: GATConv(
                        (channels, channels), channels // heads, heads=heads, dropout=dropout,
                        edge_dim=edge_dim, add_self_loops=False
                    )
                    for edge_type, edge_dim in num_edge_features.items()
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[NodeType, Tensor],
        edge_attr_dict: Dict[EdgeType, Tensor],
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        for i, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            # Removing trimming since it does not pass
            # test/external/test_node_nn.py:test_node_train_empty_graph
            # TODO: Re-introduce this.
            # Trim graph and features to only hold required data per layer:
            # if num_sampled_nodes_dict is not None:
            #     assert num_sampled_edges_dict is not None
            #     x_dict, edge_index_dict, _ = trim_to_layer(
            #         layer=i,
            #         num_sampled_nodes_per_hop=num_sampled_nodes_dict,
            #         num_sampled_edges_per_hop=num_sampled_edges_dict,
            #         x=x_dict,
            #         edge_index=edge_index_dict,
            #     )
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict


class Model(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        norm: str,
        attn_heads: int = 1,
        # List of node types to add shallow embeddings to input
        shallow_list: List[NodeType] = [],
        # ID awareness
        id_awareness: bool = False,
    ):
        super().__init__()

        # TODO consider adding an encoder for edge attributes; requires setting tf attr in
        # make_fact_dimension_graph
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        # TODO add this back. See below.
        # self.node_temporal_encoder = HeteroTemporalEncoder(
        #     node_types=[
        #         node_type for node_type in data.node_types if "time" in data[node_type]
        #     ],
        #     channels=channels,
        # )
        self.edge_temporal_encoder = HeteroTemporalEdgeEncoder(
            {e: n for e, n in data.num_edge_features.items() if "time" in data[e]}
        )
        self.gnn = HeteroGraphGAT(
            node_types=data.node_types,
            num_edge_features=data.num_edge_features,
            channels=channels,
            num_layers=num_layers,
            heads=attn_heads,
        )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
        self.embedding_dict = ModuleDict(
            {
                node: Embedding(data.num_nodes_dict[node], channels)
                for node in shallow_list
            }
        )

        self.id_awareness_emb = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        # self.temporal_edge_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        edge_attr_dict = batch.edge_attr_dict

        # TODO add this back. Need node_types to appear in batch.time_dict.
        # for node_type, rel_time in self.node_temporal_encoder(
        #     seed_time, batch.time_dict, batch.batch_dict
        # ).items():
        #     x_dict[node_type] = x_dict[node_type] + rel_time

        for edge_type, rel_time in self.edge_temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict, batch.edge_index_dict
        ).items():
            edge_attr_dict[edge_type] = edge_attr_dict[edge_type] + rel_time

        # Won't use: ignore.
        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            batch.edge_attr_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        return self.head(x_dict[entity_table][: seed_time.size(0)])

    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError(
                "id_awareness must be set True to use forward_dst_readout"
            )
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_edge_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[dst_table])
