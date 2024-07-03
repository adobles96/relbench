from itertools import permutations
import os
from typing import Any, Dict, NamedTuple, Optional, Tuple, List


import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_frame import stype
from torch_frame.config import TextEmbedderConfig
from torch_frame.data import Dataset
from torch_frame.data.stats import StatType
from torch_frame.utils import infer_df_stype
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch_geometric.utils import sort_edge_index

from relbench.data import Database, LinkTask, NodeTask, Table
from relbench.data.task_base import TaskType
from relbench.external.utils import remove_pkey_fkey, to_unix_time


def get_stype_proposal(db: Database) -> Dict[str, Dict[str, Any]]:
    r"""Propose stype for columns of a set of tables in the given database.

    Args:
        db (Database): : The database object containing a set of tables.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping table name into
            :obj:`col_to_stype` (mapping column names into inferred stypes).
    """

    inferred_col_to_stype_dict = {}
    for table_name, table in db.table_dict.items():
        inferred_col_to_stype = infer_df_stype(table.df)
        inferred_col_to_stype_dict[table_name] = inferred_col_to_stype

    return inferred_col_to_stype_dict


def make_fact_dimension_graph(
    db: Database,
    fact_tables: List[str],
    dimension_tables: List[str],
    col_to_stype_dict: Dict[str, Dict[str, stype]],
    text_embedder_cfg: Optional[TextEmbedderConfig] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[HeteroData, Dict[str, Dict[str, Dict[StatType, Any]]]]:
    data = HeteroData()  # Maybe use TemporalData ?
    col_stats_dict = dict()
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    for table_name in dimension_tables:
        table = db.table_dict[table_name]
        df = table.df
        if table.pkey_col is not None:
            assert (df[table.pkey_col].values == np.arange(len(df))).all()

        col_to_stype = col_to_stype_dict[table_name]

        # Remove pkey, fkey columns since they will not be used as input
        # feature.
        remove_pkey_fkey(col_to_stype, table)

        if len(col_to_stype) == 0:  # Add constant feature in case df is empty:
            col_to_stype = {"__const__": stype.numerical}
            # We need to add edges later, so we need to also keep the fkeys
            fkey_dict = {key: df[key] for key in table.fkey_col_to_pkey_table}
            df = pd.DataFrame({"__const__": np.ones(len(table.df)), **fkey_dict})

        path = (
            None if cache_dir is None else os.path.join(cache_dir, f"{table_name}.pt")
        )

        dataset = Dataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
        ).materialize(path=path)

        data[table_name].tf = dataset.tensor_frame
        col_stats_dict[table_name] = dataset.col_stats

        # Add time attribute:
        if table.time_col is not None:
            data[table_name].time = torch.from_numpy(
                to_unix_time(table.df[table.time_col])
            )
        # TODO I have to add fkey edges here (eg from badges to users in stack overflow).
        # Otherwise, there is nothing linking those entities.

    for table_name in fact_tables:  # aka "event" tables
        table = db.table_dict[table_name]
        df = table.df
        times = df[table.time_col]
        # attrs: drop time, pkey, fkeys and make sure to align with mask
        attrs = df.drop(
            columns=[table.time_col, table.pkey_col, *table.fkey_col_to_pkey_table.keys()]
        )

        # Add edges
        for (fkey1, fkey2) in permutations(table.fkey_col_to_pkey_table.keys(), 2):
            # draw edge between src and dst with appropriate timestamp?
            fkey1_table_name = table.fkey_col_to_pkey_table[fkey1]
            fkey2_table_name = table.fkey_col_to_pkey_table[fkey2]
            fkey1_index = df[fkey1]
            fkey2_index = df[fkey2]

            # Filter dangling foregin keys
            mask = ~fkey1_index.isna() & ~fkey2_index.isna()
            fkey1_index = torch.from_numpy(fkey1_index[mask].astype(int).values)
            fkey2_index = torch.from_numpy(fkey2_index[mask].astype(int).values)
            times = torch.from_numpy(to_unix_time(times[mask]))
            attrs = torch.from_numpy(attrs[mask])
            assert (fkey1_index < len(db.table_dict[fkey1_table_name])).all()
            assert (fkey2_index < len(db.table_dict[fkey2_table_name])).all()

            # 1 --> 2
            edge_index = torch.stack([fkey1_index, fkey2_index], dim=0)
            edge_type = (table_name, fkey1_table_name, fkey2_table_name)
            # TODO sort edges by time to speed up loader
            data[edge_type].edge_index = edge_index
            # WARNING may need to change attr name
            data[edge_type].time = times
            data[edge_type].edge_attr = attrs

            # 2 --> 1
            edge_index = torch.stack([fkey2_index, fkey1_index], dim=0)
            edge_type = (table_name, fkey2_table_name, fkey1_table_name)
            data[edge_type].edge_index = edge_index
            # WARNING may need to change attr name
            data[edge_type].time = times
            data[edge_type].edge_attr = attrs

    data.validate()
    return data, col_stats_dict


def make_pkey_fkey_graph(
    db: Database,
    col_to_stype_dict: Dict[str, Dict[str, stype]],
    text_embedder_cfg: Optional[TextEmbedderConfig] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[HeteroData, Dict[str, Dict[str, Dict[StatType, Any]]]]:
    r"""Given a :class:`Database` object, construct a heterogeneous graph with
    primary-foreign key relationships, together with the column stats of each
    table.

    Args:
        db (Database): A database object containing a set of tables.
        col_to_stype_dict (Dict[str, Dict[str, stype]]): Column to stype for
            each table.
        text_embedder_cfg (TextEmbedderConfig): Text embedder config.
        cache_dir (str, optional): A directory for storing materialized tensor
            frames. If specified, we will either cache the file or use the
            cached file. If not specified, we will not use cached file and
            re-process everything from scratch without saving the cache.

    Returns:
        HeteroData: The heterogeneous :class:`PyG` object with
            :class:`TensorFrame` feature.
    """
    data = HeteroData()
    col_stats_dict = dict()
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    for table_name, table in db.table_dict.items():
        # Materialize the tables into tensor frames:
        df = table.df
        # Ensure that pkey is consecutive.
        if table.pkey_col is not None:
            assert (df[table.pkey_col].values == np.arange(len(df))).all()

        col_to_stype = col_to_stype_dict[table_name]

        # Remove pkey, fkey columns since they will not be used as input
        # feature.
        remove_pkey_fkey(col_to_stype, table)

        if len(col_to_stype) == 0:  # Add constant feature in case df is empty:
            col_to_stype = {"__const__": stype.numerical}
            # We need to add edges later, so we need to also keep the fkeys
            fkey_dict = {key: df[key] for key in table.fkey_col_to_pkey_table}
            df = pd.DataFrame({"__const__": np.ones(len(table.df)), **fkey_dict})

        path = (
            None if cache_dir is None else os.path.join(cache_dir, f"{table_name}.pt")
        )

        dataset = Dataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
        ).materialize(path=path)

        data[table_name].tf = dataset.tensor_frame
        col_stats_dict[table_name] = dataset.col_stats

        # Add time attribute:
        if table.time_col is not None:
            data[table_name].time = torch.from_numpy(
                to_unix_time(table.df[table.time_col])
            )

        # Add edges:
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            pkey_index = df[fkey_name]
            # Filter out dangling foreign keys
            mask = ~pkey_index.isna()
            fkey_index = torch.arange(len(pkey_index))
            # Filter dangling foreign keys:
            pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
            fkey_index = fkey_index[torch.from_numpy(mask.values)]
            # Ensure no dangling fkeys
            assert (pkey_index < len(db.table_dict[pkey_table_name])).all()

            # fkey -> pkey edges
            edge_index = torch.stack([fkey_index, pkey_index], dim=0)
            edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

            # pkey -> fkey edges.
            # "rev_" is added so that PyG loader recognizes the reverse edges
            edge_index = torch.stack([pkey_index, fkey_index], dim=0)
            edge_type = (pkey_table_name, f"rev_f2p_{fkey_name}", table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

    data.validate()

    return data, col_stats_dict


class AttachTargetTransform:
    r"""Adds the target label to the heterogeneous mini-batch.
    The batch consists of disjoins subgraphs loaded via temporal sampling.
    The same input node can occur twice with different timestamps, and thus
    different subgraphs and labels. Hence labels cannot be stored in the graph
    object directly, and must be attached to the batch after the batch is
    created."""

    def __init__(self, entity: str, target: Tensor):
        self.entity = entity
        self.target = target

    def __call__(self, batch: HeteroData) -> HeteroData:
        batch[self.entity].y = self.target[batch[self.entity].input_id]
        return batch


class NodeTrainTableInput(NamedTuple):
    nodes: Tuple[NodeType, Tensor]
    time: Optional[Tensor]
    target: Optional[Tensor]
    transform: Optional[AttachTargetTransform]


def get_node_train_table_input(
    table: Table,
    task: NodeTask,
    multilabel: bool = False,
) -> NodeTrainTableInput:
    nodes = torch.from_numpy(table.df[task.entity_col].astype(int).values)

    time: Optional[Tensor] = None
    if table.time_col is not None:
        time = torch.from_numpy(to_unix_time(table.df[table.time_col]))

    target: Optional[Tensor] = None
    transform: Optional[AttachTargetTransform] = None
    if task.target_col in table.df:
        target_type = float
        if task.task_type == "multiclass_classification":
            target_type = int
        if task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            target = torch.from_numpy(np.stack(table.df[task.target_col].values))
        else:
            target = torch.from_numpy(
                table.df[task.target_col].values.astype(target_type)
            )
        transform = AttachTargetTransform(task.entity_table, target)

    return NodeTrainTableInput(
        nodes=(task.entity_table, nodes),
        time=time,
        target=target,
        transform=transform,
    )


class AttachTemporalTargetTransform:
    r"""Adds the target label to the heterogeneous mini-batch.
    The batch consists of disjoins subgraphs loaded via temporal sampling.
    The same input node can occur twice with different timestamps, and thus
    different subgraphs and labels. Hence labels cannot be stored in the graph
    object directly, and must be attached to the batch after the batch is
    created."""

    def __init__(self, entity: str, target_dict: Dict[str, Tensor]):
        self.entity = entity
        self.target_dict = target_dict

    def __call__(self, batch: HeteroData, target_key: str) -> HeteroData:
        batch[self.entity].y = self.target_dict[target_key][batch[self.entity].input_id]
        return batch


class NodeTemporalTrainTableInput(NamedTuple):
    nodes: Tuple[NodeType, Tensor]
    time: Optional[Tensor]
    target: Optional[Tensor]
    transform: Optional[AttachTemporalTargetTransform]
    previous_times: Optional[Tensor]


def get_temporal_node_train_table_input(
    table: Table,
    task: NodeTask,
    num_ar: int,   
    multilabel: bool = False, 
) -> NodeTrainTableInput:
    nodes = torch.from_numpy(table.df[task.entity_col].astype(int).values)

    time: Optional[Tensor] = None
    if table.time_col is not None:
        time = torch.from_numpy(to_unix_time(table.df[table.time_col]))

    target: Optional[Tensor] = None
    transform: Optional[AttachTemporalTargetTransform] = None
    if task.target_col in table.df:
        target_type = float
        if task.task_type == "multiclass_classification":
            target_type = int
        if task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            target = torch.from_numpy(np.stack(table.df[task.target_col].values))
        else:
            target = torch.from_numpy(
                table.df[task.target_col].values.astype(target_type)
            )


        target_dict = {'root': target}

        previous_times = []
        for i in range(1, num_ar + 1):
            prev = torch.from_numpy(to_unix_time(table.df[table.time_col] - i*task.timedelta))
            previous_times.append(prev)
            target_dict[f'AR_{i}'] =  torch.from_numpy(
                table.df[f'AR_{i}'].values.astype(target_type)
            )

        previous_times = torch.stack(previous_times, dim=1)
        
        transform = AttachTemporalTargetTransform(task.entity_table, target_dict)

    return NodeTemporalTrainTableInput(
        nodes=(task.entity_table, nodes),
        time=time,
        target=target,
        transform=transform,
        previous_times=previous_times,
    )

class LinkTrainTableInput(NamedTuple):
    r"""Trainining table input for link prediction.

    - src_nodes is a Tensor of source node indices.
    - dst_nodes is PyTorch sparse tensor in csr format.
        dst_nodes[src_node_idx] gives a tensor of destination node
        indices for src_node_idx.
    - num_dst_nodes is the total number of destination nodes.
        (used to perform negative sampling).
    - src_time is a Tensor of time for src_nodes
    """

    src_nodes: Tuple[NodeType, Tensor]
    dst_nodes: Tuple[NodeType, Tensor]
    num_dst_nodes: int
    src_time: Optional[Tensor]


def get_link_train_table_input(
    table: Table,
    task: LinkTask,
) -> LinkTrainTableInput:
    src_node_idx: Tensor = torch.from_numpy(
        table.df[task.src_entity_col].astype(int).values
    )
    exploded = table.df[task.dst_entity_col].explode()
    coo_indices = torch.from_numpy(
        np.stack([exploded.index.values, exploded.values.astype(int)])
    )
    sparse_coo = torch.sparse_coo_tensor(
        coo_indices,
        torch.ones(coo_indices.size(1), dtype=bool),
        (len(src_node_idx), task.num_dst_nodes),
    )
    dst_node_indices = sparse_coo.to_sparse_csr()

    time: Optional[Tensor] = None
    if table.time_col is not None:
        time = torch.from_numpy(to_unix_time(table.df[table.time_col]))

    return LinkTrainTableInput(
        src_nodes=(task.src_entity_table, src_node_idx),
        dst_nodes=(task.dst_entity_table, dst_node_indices),
        num_dst_nodes=task.num_dst_nodes,
        src_time=time,
    )
