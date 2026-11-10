import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TypeVar

import numpy as np
import pandas as pd
import torch
from sklearn.base import TransformerMixin
from typing_extensions import deprecated

from tabdiff import utils as utils
from tabdiff.datasets.processors import split_num_cat_target
from tabdiff.utils import (
    Dequantizer,
    IdentityTransformer,
    TaskType,
    Transformations,
    get_categories,
    load_json,
    read_pure_data,
    Dataset,
    transform_dataset,
)

# Define overloaded type for clarity in usage.
CatArray = TypeVar("CatArray", bound=np.ndarray)
NumArray = TypeVar("NumArray", bound=np.ndarray)


@deprecated(
    "Preprocessing from data-path is deprecated, use `encode_dataset` and `make_dataset`"
)
def preprocess(
    dataset_path,
    y_only=False,
    task_type="binclass",
    inverse=False,
    concat=True,
    col_to_remove=None,
    info: Dict[str, Any] = None,
    transforms: Transformations = None,
    fitted_transformations: Optional[
        Tuple[
            Optional[TransformerMixin],
            Optional[TransformerMixin],
            Optional[TransformerMixin],
        ]
    ] = None,
    data: Dict[str, pd.DataFrame] = None,
):
    """

    Args:
        dataset_path ():
        y_only ():
        dequant_dist ():
        int_dequant_factor ():
        task_type ():
        inverse ():
        cat_encoding ():
        concat ():
        col_to_remove ():
        info (dict, *, None): Metadata dictionary when constructing dataset from alternative source.
        transforms (Transformations): Transformations to be applied to the generative model.
        fitted_transformations: Optional pre-fitted transforms to encode dataset using pre-fitted transformations.
        data (Dict[str,pd.DataFrame], *, None): Optional DataFrame dictionary containing pairings of different
            splits of data. I.e., mapping existing data back to real (e.g. in case of perturbation of generated ata).

    Returns:

    """

    # Make-dataset, yields pre-encoded dataset object
    dataset = make_dataset(
        data_path=dataset_path,
        T=transforms,
        task_type=task_type,
        change_val=False,
        concat=concat,
        y_only=y_only,
        col_to_remove=col_to_remove,
        info=info,
        data=data,
        fitted_transformations=fitted_transformations,
    )
    # Applies transforms from data-domain to value domain
    return encode_dataset(
        dataset,
        inverse,
        transforms,
    )


def encode_dataset(
    dataset: Dataset,
    inverse: bool = False,
    transforms: Optional[Transformations] = None,
) -> Dataset:
    """
    Apply transformations on loaded dataset information.
    Args:
        dataset ():
        inverse ():
        transforms ():

    Returns:

    """
    if transforms.cat_encoding is None:
        X_num = dataset.X_num
        X_cat = dataset.X_cat

        X_train_num, X_test_num = X_num["train"], X_num.get("test", None)
        X_train_cat, X_test_cat = X_cat["train"], X_cat.get("test", None)

        categories = get_categories(X_train_cat)
        d_numerical = X_train_num.shape[1]

        X_num = (X_train_num, X_test_num)
        X_cat = (X_train_cat, X_test_cat)

        if inverse:
            identity = IdentityTransformer()
            num_transform = (
                dataset.num_transform if dataset.num_transform is not None else identity
            )
            int_transform = (
                dataset.int_transform if dataset.int_transform is not None else identity
            )
            cat_transform = (
                dataset.cat_transform if dataset.cat_transform is not None else identity
            )

            return (
                X_num,
                X_cat,
                categories,
                d_numerical,
                num_transform,
                int_transform,
                cat_transform,
            )
        else:
            return X_num, X_cat, categories, d_numerical
    else:
        return dataset


def prefix_hstack_target(X, y):
    """
    Helper to horizontally stack array (target column) **in front** of the features.

    Notes:
        This method pre-pends the target column to the input-space.
    Args:
        X (np.ndarray[float | int]):
        y (np.ndarray[float | int]):

    Returns:
        np.ndarray[float | int]: Concatenated ndarray with target infront of features
    """
    if X is None:
        return y.reshape(-1, 1)
    # Do unsqueeze, then stack as column in 2nd axis.
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


def make_dataset(
    data_path: str,
    T: Transformations,
    task_type,
    change_val: bool,
    concat=True,
    y_only=False,
    col_to_remove=None,
    info: Dict[str, Any] = None,
    data: Dict[str, pd.DataFrame] = None,
    fitted_transformations: Optional[
        Tuple[
            Optional[TransformerMixin],
            Optional[TransformerMixin],
            Optional[TransformerMixin],
        ]
    ] = None,
) -> Dataset:
    # Load meta-data and default meta-data in case a dataframe is readily provided.
    info = info or load_json(os.path.join(data_path, "info.json"))
    split_info = info
    int_col_idx_wrt_num = info["int_col_idx_wrt_num"]

    # regression
    # Here we pray to god that the info object is true.
    # also, why are we loading data during the make_dataset call.....
    # my god this codebase is spaghet.
    X_cat, X_num, y = extract_raw_data(data_path, data, info)

    if data is not None:

        for split in X_num.keys():
            X_num_t = X_num[split]
            X_cat_t = X_cat[split]
            y_t = y[split]

            if y_only:
                # If imputation, only get nothgin
                X_num_t = X_num_t[:, :0]
                X_cat_t = X_cat_t[:, :0]
            if X_num_t is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                if concat:
                    X_cat_t = prefix_hstack_target(X_cat_t, y_t)
                X_cat[split] = X_cat_t
            if y is not None:
                y[split] = y_t
        D = utils.Dataset(
            X_num,
            X_cat,
            y,
            int_col_idx_wrt_num,
            task_type=TaskType(split_info["task_type"]),
            n_classes=split_info.get("n_classes"),
            y_info={},
        )
    else:
        # Old way of hanlding, needed for now.
        if task_type == "binclass" or task_type == "multiclass":
            # classification
            for split in ["train", "test"]:
                # Reads from pre-generated npy data
                X_num_t, X_cat_t, y_t, split_info, rel_deletion_indx = read_pure_data(
                    data_path,
                    split,
                    col_to_remove=col_to_remove,
                    metadata=info,
                )
                if y_only:
                    # If imputation, only get nothgin
                    X_num_t = X_num_t[:, :0]
                    X_cat_t = X_cat_t[:, :0]
                if X_num is not None:
                    # Get split data
                    X_num[split] = X_num_t
                if X_cat is not None:
                    if concat:
                        X_cat_t = prefix_hstack_target(X_cat_t, y_t)
                    X_cat[split] = X_cat_t
                if y is not None:
                    y[split] = y_t
            if rel_deletion_indx is not None:
                # Update after removal...
                if rel_deletion_indx in int_col_idx_wrt_num:
                    int_col_idx_wrt_num = [
                        (indx - 1) if indx > rel_deletion_indx else indx
                        for indx in int_col_idx_wrt_num
                        if indx != rel_deletion_indx
                    ]
        else:
            # Regression
            for split in ["train", "test"]:
                X_num_t, X_cat_t, y_t, split_info, rel_deletion_indx = read_pure_data(
                    data_path,
                    split,
                    col_to_remove=col_to_remove,
                    metadata=info,
                )
                # IF IMPUTING, we set the arrays to retain the data's shape, but remove all columns
                if y_only:
                    X_num_t = X_num_t[:, :0]
                    X_cat_t = X_cat_t[:, :0]

                # Second, concat the target to the tensor
                if X_num is not None:
                    if concat:
                        X_num_t = prefix_hstack_target(X_num_t, y_t)
                    X_num[split] = X_num_t
                if X_cat is not None:
                    X_cat[split] = X_cat_t
                if y is not None:
                    y[split] = y_t
            if rel_deletion_indx is not None:
                # Update after removal...
                if rel_deletion_indx in int_col_idx_wrt_num:
                    int_col_idx_wrt_num = [
                        (indx - 1) if indx > rel_deletion_indx else indx
                        for indx in int_col_idx_wrt_num
                        if indx != rel_deletion_indx
                    ]
        if y_only:
            int_col_idx_wrt_num = []
        D = utils.Dataset(
            X_num,
            X_cat,
            y,
            int_col_idx_wrt_num,
            y_info={},
            task_type=TaskType(split_info["task_type"]),
            n_classes=split_info.get("n_classes"),
        )

        if change_val:
            D = utils.change_val(D)

    return transform_dataset(
        D,
        T,
        None,
        fitted_transforms=fitted_transformations,
    )


def extract_raw_data(
    data_path,
    data: Optional[Dict[str, pd.DataFrame]],
    info: Dict[str, Any],
    col_to_remove=None,
) -> Tuple[Optional[CatArray], Optional[NumArray], np.ndarray]:
    """
    Method to retrieve data, or extract from provided dataframe.
    Args:
        data_path (Path):
        data ():
        info ():

    Returns:

    """
    if data is not None and isinstance(data, dict):
        assert isinstance(
            info, dict
        ), "Without provided metadata, cannot re-construct dataset"
        assert sorted(info.get("column_names", [])) == sorted(
            list(data.values())[0].columns
        ), (
            "Columns must match between meta-data object and "
            "provided dataset. (Skyline only)"
        )

        # Expects columns to be in order of the original dataset on disk/after preparation
        cat_column_indices = sorted(info.get("cat_col_idx", []))
        target_column_indices = sorted(info.get("target_col_idx", []))

        # Expects columns to be in order of the original dataset on disk/after preparation
        #   Note, numerical columsn are those that are not categorical.
        num_column_indices = set(range(len(info.get("idx_mapping", dict()))))
        num_column_indices.difference_update(cat_column_indices)
        num_column_indices.difference_update(target_column_indices)
        num_column_indices = sorted(list(num_column_indices))

        X_cat = {
            key: (
                value.iloc[:, cat_column_indices].to_numpy()
                if cat_column_indices
                else None
            )
            for key, value in data.items()
        }
        X_num = {
            key: (
                value.iloc[:, num_column_indices].to_numpy()
                if num_column_indices
                else None
            )
            for key, value in data.items()
        }
        y = {
            key: (
                value.iloc[:, target_column_indices].to_numpy()
                if target_column_indices
                else None
            )
            for key, value in data.items()
        }
    else:
        X_cat = (
            {} if os.path.exists(os.path.join(data_path, "X_cat_train.npy")) else None
        )
        X_num = (
            {} if os.path.exists(os.path.join(data_path, "X_num_train.npy")) else None
        )
        y = {} if os.path.exists(os.path.join(data_path, "y_train.npy")) else None

    return X_cat, X_num, y


def concat_encoded_data(
    X_cat: np.ndarray | None, X_num: np.ndarray | None
) -> torch.FloatTensor:
    """
    Concatenate encoded data.
    Args:
        X_cat (): Optional encoded categorical data.
        X_num (): Optional encoded numerical data.
        is_train ():

    Returns:

    """

    X_num_torch, X_cat_torch = (
        torch.tensor(X_num).float(),
        torch.tensor(X_cat).float(),
    )
    tensors = []
    if isinstance(X_num_torch, torch.Tensor):
        tensors.append(X_num_torch)

    if isinstance(X_cat_torch, torch.Tensor):
        tensors.append(X_cat_torch)
    X: torch.FloatTensor = torch.cat(tensors, dim=1).to(torch.float)
    return X
