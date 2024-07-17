import argparse
from typing import Dict

import numpy as np
import pandas as pd
import torch
from scipy.stats import mode
from torch_geometric.seed import seed_everything

from relbench.base import Dataset, NodeTask, Table, TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-stack")
parser.add_argument("--task", type=str, default="user-engagement")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_ar_labels", type=int, default=5)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: NodeTask = get_task(args.dataset, args.task, download=True)

train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test")


def evaluate(train_table: Table,
             pred_table: Table,
            name: str,
             val_table: Table = None,
             test_table: Table = None,
             ) -> Dict[str, float]:
    is_test = task.target_col not in pred_table.df
    if name == "global_zero":
        pred = np.zeros(len(pred_table))
    elif name == "global_mean":
        mean = train_table.df[task.target_col].astype(float).values.mean()
        pred = np.ones(len(pred_table)) * mean
    elif name == "global_median":
        median = np.median(train_table.df[task.target_col].astype(float).values)
        pred = np.ones(len(pred_table)) * median
    elif name == "entity_mean":
        fkey = list(train_table.fkey_col_to_pkey_table.keys())[0]
        df = train_table.df.groupby(fkey).agg({task.target_col: "mean"})
        df.rename(columns={task.target_col: "__target__"}, inplace=True)
        df = pred_table.df.merge(df, how="left", on=fkey)
        pred = df["__target__"].fillna(0).astype(float).values
    elif name == "entity_median":
        fkey = list(train_table.fkey_col_to_pkey_table.keys())[0]
        df = train_table.df.groupby(fkey).agg({task.target_col: "median"})
        df.rename(columns={task.target_col: "__target__"}, inplace=True)
        df = pred_table.df.merge(df, how="left", on=fkey)
        pred = df["__target__"].fillna(0).astype(float).values
    elif name == "random":
        pred = np.random.rand(len(pred_table))
    elif name == "majority":
        past_target = train_table.df[task.target_col].astype(int)
        majority_label = int(past_target.mode().iloc[0])
        pred = torch.full((len(pred_table),), fill_value=majority_label)
    elif name == "majority_multilabel":
        past_target = train_table.df[task.target_col]
        majority = mode(np.stack(past_target.values), axis=0).mode[0]
        pred = np.stack([majority] * len(pred_table.df))
    elif name == "random_multilabel":
        num_labels = train_table.df[task.target_col].values[0].shape[0]
        pred = np.random.rand(len(pred_table), num_labels)
    elif "ar" in name:
        ar_label_cols = []

        ### Adding AR labels into train/val/test_table
        whole_df = pd.concat([train_table.df, val_table.df, test_table.df], axis=0)
        num_ar_labels = args.num_ar_labels

        sorted_unique_times = np.sort(whole_df[train_table.time_col].unique())
        timedelta = sorted_unique_times[1:] - sorted_unique_times[:-1]

        TIME_IDX_COL = "time_idx"
        time_df = pd.DataFrame(
            {
                task.time_col: sorted_unique_times,
                "time_idx": np.arange(len(sorted_unique_times)),
            }
        )

        whole_df = whole_df.merge(time_df, how="left", on=task.time_col)
        whole_df.drop(task.time_col, axis=1, inplace=True)
        # Shift timestamp of whole_df iteratively and join it with train/val/test_table
        for i in range(1, num_ar_labels + 1):
            whole_df_shifted = whole_df.copy(deep=True)
            # Shift time index by i
            whole_df_shifted[TIME_IDX_COL] += i
            # Map time index back to datetime timestamp
            whole_df_shifted = whole_df_shifted.merge(time_df, how="inner", on=TIME_IDX_COL)
            whole_df_shifted.drop(TIME_IDX_COL, axis=1, inplace=True)
            ar_label = f"AR_{i}"
            ar_label_cols.append(ar_label)
            whole_df_shifted.rename(columns={task.target_col: ar_label}, inplace=True)

            for table in [train_table, val_table, test_table]:
                table.df = table.df.merge(
                    whole_df_shifted, how="left", on=(task.entity_col, task.time_col)
                )

        # get all columns of test table with "AR_" prefix
        ar_cols = [col for col in test_table.df.columns if "AR_" in col]
        past = test_table.df[ar_cols].values

        if name == "ar_classify":
            # get most common ar label, ignoring NaNs
            pred = mode(past, axis=1, nan_policy="omit", keepdims=True)[0]

        elif name == "ar_regression":
            # get mean of ar labels, ignoring NaNs
            pred = np.nanmean(past, axis=1)
            # replace nans with mean of non-nan values
            pred = np.where(np.isnan(pred), np.nanmean(pred), pred)

    else:
        raise ValueError("Unknown eval name called {name}.")
    return task.evaluate(pred, None if is_test else pred_table)


trainval_table_df = pd.concat([train_table.df, val_table.df], axis=0)
trainval_table = Table(
    df=trainval_table_df,
    fkey_col_to_pkey_table=train_table.fkey_col_to_pkey_table,
    pkey_col=train_table.pkey_col,
    time_col=train_table.time_col,
)

full_test_table = task._full_test_table

if task.task_type == TaskType.REGRESSION:
    eval_name_list = [
        "global_zero",
        "global_mean",
        "global_median",
        "entity_mean",
        "entity_median",
        "ar_regression",
    ]

    for name in eval_name_list:
        if not name.startswith("ar"):
            train_metrics = evaluate(train_table, train_table, name=name, val_table=val_table, test_table=train_table)
            val_metrics = evaluate(train_table, val_table, name=name, val_table=val_table, test_table=val_table)
        else:
            train_metrics = None
            val_metrics = None
        test_metrics = evaluate(trainval_table, test_table, name=name, val_table=val_table, test_table=full_test_table)
        print(f"{name}:")
        print(f"Train: {train_metrics}")
        print(f"Val: {val_metrics}")
        print(f"Test: {test_metrics}")



elif task.task_type == TaskType.BINARY_CLASSIFICATION:
    eval_name_list = ["random", "majority", "ar_classify"]
    for name in eval_name_list:
        if not name.startswith("ar"):
            train_metrics = evaluate(train_table, train_table, name=name, val_table=val_table, test_table=train_table)
            val_metrics = evaluate(train_table, val_table, name=name, val_table=val_table, test_table=val_table)
        else:
            train_metrics = None
            val_metrics = None
        test_metrics = evaluate(trainval_table, test_table, name=name, val_table=val_table, test_table=full_test_table)
        print(f"{name}:")
        print(f"Train: {train_metrics}")
        print(f"Val: {val_metrics}")
        print(f"Test: {test_metrics}")


elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    eval_name_list = ["random_multilabel", "majority_multilabel"]
    for name in eval_name_list:
        train_metrics = evaluate(train_table, train_table, name=name)
        val_metrics = evaluate(train_table, val_table, name=name)
        test_metrics = evaluate(trainval_table, test_table, name=name)
        print(f"{name}:")
        print(f"Train: {train_metrics}")
        print(f"Val: {val_metrics}")
        print(f"Test: {test_metrics}")
