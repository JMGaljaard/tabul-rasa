from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch


def inverse_data_transform(
    info,
    syn_data,
    cat_inverse,
    int_inverse,
    num_inverse,
    y_only=False,
):
    """
    Maps data-tensor from model's domain to value domain
    Args:
        info ():
        syn_data ():
        cat_inverse ():
        int_inverse ():
        num_inverse ():
        y_only ():

    Returns:

    """
    if y_only:
        if info["task_type"] == "binclass":
            syn_data = cat_inverse(syn_data)
        else:
            syn_data = num_inverse(syn_data)
        syn_df = pd.DataFrame()
        syn_df[info["column_names"][info["target_col_idx"][0]]] = syn_data[:, 0]
    else:
        # TODO: FIgure out thsi dataset's issue
        syn_num, syn_cat, syn_target = split_num_cat_target(
            syn_data, info, num_inverse, int_inverse, cat_inverse
        )
        syn_df = recover_data(syn_num, syn_cat, syn_target, info)

        idx_name_mapping = info["idx_name_mapping"]
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

        syn_df.rename(columns=idx_name_mapping, inplace=True)
    return syn_df


@torch.no_grad()
def split_num_cat_target(
    syn_data: torch.Tensor, info, num_inverse, int_inverse, cat_inverse
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    task_type = info["task_type"]

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == "regression":
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    syn_num = syn_data[:, :n_num_feat]
    syn_cat = syn_data[:, n_num_feat:]

    syn_num = num_inverse(syn_num).astype(np.float32)
    syn_num = int_inverse(syn_num).astype(np.float32)
    syn_cat = cat_inverse(syn_cat)

    if info["task_type"] == "regression":
        syn_target = syn_num[:, : len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx) :]

    else:
        syn_target = syn_cat[:, : len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx) :]

    return syn_num, syn_cat, syn_target


def recover_data(
    syn_num: np.ndarray,
    syn_cat: np.ndarray,
    syn_target: np.ndarray,
    info: Dict[str, Any],
):
    """
    Given split data, reconstruct a dataframe.
    Args:
        syn_num ():
        syn_cat ():
        syn_target ():
        info ():

    Returns:
        pd.DataFrame
    """

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    idx_mapping = info["idx_mapping"]
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()
    num_col_count = len(num_col_idx)
    cat_col_count = len(cat_col_idx)
    # For each # DATA index
    for dataframe_index in range(
        len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)
    ):
        # For 'each' encoded index.
        joint_encoded_index = int(idx_mapping[dataframe_index])
        if dataframe_index in set(num_col_idx):
            num_encoded_index = joint_encoded_index
            syn_df[dataframe_index] = syn_num[:, num_encoded_index]

        elif dataframe_index in set(cat_col_idx):
            # Subtract number of numerical columns, as categoricals are
            cat_encoded_index = joint_encoded_index - num_col_count
            syn_df[dataframe_index] = syn_cat[:, cat_encoded_index]

        else:
            # Note, this is ALWAYS ZERO.
            # Note, the 'index' assumes that the target is always the last column,
            # but, during the pre-processing the column is prepended to either the
            # numerical or categorical features. WTH.
            target_encoded_index = joint_encoded_index - (num_col_count + cat_col_count)
            syn_df[dataframe_index] = syn_target[:, target_encoded_index]

    return syn_df
