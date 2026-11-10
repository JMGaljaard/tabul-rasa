import os
import json
from typing import Any, Dict

import numpy as np
import pandas as pd
from pathlib import Path


def get_column_name_mapping(
    data_df, num_col_idx, cat_col_idx, target_col_idx, column_names=None
):

    if not column_names:
        column_names = np.array(data_df.columns.tolist())

    idx_mapping = {}

    curr_num_idx = 0
    curr_cat_idx = len(num_col_idx)
    curr_target_idx = curr_cat_idx + (len(cat_col_idx) if cat_col_idx else 0)

    for idx in range(len(column_names)):

        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num_idx
            curr_num_idx += 1
        elif cat_col_idx and idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat_idx
            curr_cat_idx += 1
        else:
            idx_mapping[int(idx)] = curr_target_idx
            curr_target_idx += 1

    inverse_idx_mapping = {}
    for k, v in idx_mapping.items():
        inverse_idx_mapping[int(v)] = k

    idx_name_mapping = {}

    for i in range(len(column_names)):
        idx_name_mapping[int(i)] = column_names[i]

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def train_val_test_split(data_df, cat_columns, num_train=0, num_test=0, seed=1234):
    total_num = data_df.shape[0]
    idx = np.arange(total_num)

    while True:
        np.random.seed(seed)
        np.random.shuffle(idx)

        train_idx = idx[:num_train]
        test_idx = idx[-num_test:]

        train_df = data_df.loc[train_idx]
        test_df = data_df.loc[test_idx]

        flag = 0
        if not cat_columns:
            break
        for i in cat_columns:
            if len(set(train_df[i])) != len(set(data_df[i])):
                flag = 1
                break

        if flag == 0:
            break
        else:
            seed += 1

    return train_df, test_df, seed


def split_dataset(df, info, cat_columns, seed: int = 1234):
    """
    Splits the dataset into train/test/val according to dataset rules.
    Supports:
      - DCR datasets: 50/50 split for train/test
      - Provided test_path and val_path
      - Default 90/10 train/test split
    Returns: train_df, test_df, val_df
    """
    has_val = bool(info.get("val_path"))
    val_df = pd.DataFrame(columns=df.columns).astype(df.dtypes)
    test_df = pd.DataFrame(columns=df.columns).astype(df.dtypes)
    # Load test and val if provided
    if info.get("test_path"):
        test_df = pd.read_csv(info["test_path"], header=info.get("header", 0))

    if has_val:
        val_df = pd.read_csv(info["val_path"], header=info.get("header", 0))

    train_df = df.copy()

    def cleanup_whitespaces(inner_df):
        inner_df.replace(r"^ +", r"", regex=True, inplace=True)
        inner_df.columns = train_df.columns.str.strip()
        return inner_df

    # Standard 90/10 split or 50/50 for DCR
    # For "dcr" datasets, override with 50/50 split
    complete_df = pd.concat(
        list(map(cleanup_whitespaces, [train_df, test_df, val_df])), axis=0
    )
    if "dcr" in info["name"]:

        num_train = int(len(complete_df) * 0.5)
        num_test = len(complete_df) - num_train
        complete_df.rename(columns=lambda x: x, inplace=True)
        train_df, test_df, _ = train_val_test_split(
            complete_df,
            cat_columns,
            num_train,
            num_test,
            seed=seed,
        )
    else:
        num_train = int(len(df) * (0.5 if "dcr" in info["name"] else 0.9))
        num_test = len(df) - num_train
        train_df, test_df, val_df = train_val_test_split(
            complete_df,
            cat_columns,
            num_train,
            num_test,
            seed=seed,
        )

    return train_df, test_df, val_df


def load_dataset(info):
    data_path = info["data_path"]
    if info.get("file_type", "csv") == "csv":
        df = pd.read_csv(data_path, header=info.get("header", 0))
    elif info["file_type"] == "xls":
        df = pd.read_excel(data_path, sheet_name="Data", header=1)
        if "ID" in df.columns:
            df = df.drop("ID", axis=1)
    else:
        raise ValueError(f"Unsupported file type: {info.get('file_type')}")
    return df


# ---------------------- Missing Values ---------------------- #


def handle_missing_values(df, num_columns, cat_columns):
    for col in num_columns:
        df[col] = df[col].replace(["?", " ?"], np.nan)
    for col in cat_columns:
        df[col] = df[col].replace(["?"], "nan")
    return df


# ---------------------- Integer Detection ---------------------- #


def detect_integer_columns(df, num_col_idx, column_names, idx_name_mapping):
    int_columns, int_col_idx, int_col_idx_wrt_num = [], [], []
    for i, col_idx in enumerate(num_col_idx):
        col = column_names[col_idx]
        if (df[col] % 1 == 0).all():
            int_columns.append(col)
            int_col_idx.append(idx_name_mapping[col_idx])
            int_col_idx_wrt_num.append(i)
    return int_columns, int_col_idx, int_col_idx_wrt_num


# ---------------------- Save Data ---------------------- #
def save_dataset(
    train_df,
    test_df,
    val_df,
    num_columns,
    cat_columns,
    target_columns,
    name,
    has_val,
):
    save_dir = Path("data") / name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Numpy arrays
    np.save(
        save_dir / "X_num_train.npy",
        train_df[num_columns].to_numpy().astype(np.float32),
    )
    np.save(save_dir / "X_cat_train.npy", train_df[cat_columns].to_numpy())
    np.save(save_dir / "y_train.npy", train_df[target_columns].to_numpy())

    np.save(
        save_dir / "X_num_test.npy", test_df[num_columns].to_numpy().astype(np.float32)
    )
    np.save(save_dir / "X_cat_test.npy", test_df[cat_columns].to_numpy())
    np.save(save_dir / "y_test.npy", test_df[target_columns].to_numpy())

    if has_val:
        np.save(
            save_dir / "X_num_val.npy",
            val_df[num_columns].to_numpy().astype(np.float32),
        )
        np.save(save_dir / "X_cat_val.npy", val_df[cat_columns].to_numpy())
        np.save(save_dir / "y_val.npy", val_df[target_columns].to_numpy())

    # CSV files
    for df_name, df in zip(["train", "test", "val"], [train_df, test_df, val_df]):
        if df_name == "val" and not has_val:
            continue
        df.to_csv(save_dir / f"{df_name}.csv", index=False)


# ---------------------- Metadata Update ---------------------- #
def update_metadata(
    info,
    train_df,
    num_columns,
    cat_columns,
    target_columns,
    idx_mapping,
    inverse_idx_mapping,
    idx_name_mapping,
):
    info["train_num"] = len(train_df)
    info["num_col_idx"] = [i for i in range(len(num_columns))]
    info["cat_col_idx"] = [len(num_columns) + i for i in range(len(cat_columns))]
    info["target_col_idx"] = [len(num_columns) + len(cat_columns)]
    info["idx_mapping"] = idx_mapping
    info["inverse_idx_mapping"] = inverse_idx_mapping
    info["idx_name_mapping"] = idx_name_mapping

    metadata = {"columns": {}}
    for i in info["num_col_idx"]:
        metadata["columns"][i] = {
            "sdtype": "numerical",
            "computer_representation": "Float",
        }
    for i in info["cat_col_idx"]:
        metadata["columns"][i] = {"sdtype": "categorical"}
    for i in info["target_col_idx"]:
        if info["task_type"] == "regression":
            metadata["columns"][i] = {
                "sdtype": "numerical",
                "computer_representation": "Float",
            }
        else:
            metadata["columns"][i] = {"sdtype": "categorical"}
    info["metadata"] = metadata

    save_dir = Path("data") / info["name"]
    with open(save_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    return info


# ---------------------- Main Processing ---------------------- #


def process_data(
    name: str,
    info: Dict[str, Any],
    data_df: pd.DataFrame,
    seed: int = 1234,
):
    """
    Retrieve pre-processed dataset into train, test, and, validation set.
    Args:
        name ():
        info ():
        data_df ():
        seed ():

    Returns:

    """

    # Column info
    column_names = info.get("column_names") or data_df.columns.tolist()
    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]
    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]
    target_columns = [column_names[i] for i in target_col_idx]

    # Split
    train_df, test_df, val_df = split_dataset(
        data_df,
        info,
        cat_columns,
        seed=seed,
    )

    # Missing values
    train_df = handle_missing_values(train_df, num_columns, cat_columns)
    test_df = handle_missing_values(test_df, num_columns, cat_columns)
    val_df = handle_missing_values(val_df, num_columns, cat_columns)

    # Column mappings
    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(
        data_df, num_col_idx, cat_col_idx, target_col_idx, column_names
    )

    # Integer detection
    int_columns, int_col_idx, int_col_idx_wrt_num = detect_integer_columns(
        pd.concat([train_df, test_df, val_df]),
        num_col_idx,
        column_names,
        idx_name_mapping,
    )
    info.update(
        {
            "int_columns": int_columns,
            "int_col_idx": int_col_idx,
            "int_col_idx_wrt_num": int_col_idx_wrt_num,
        }
    )

    # Update metadata
    info = update_metadata(
        info,
        train_df,
        num_columns,
        cat_columns,
        target_columns,
        idx_mapping,
        inverse_idx_mapping,
        idx_name_mapping,
    )

    return train_df, test_df, val_df, info
