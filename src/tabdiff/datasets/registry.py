import json
from pathlib import Path
import numpy as np
import omegaconf
import pandas as pd

from .preprocessing import (
    detect_integer_columns,
    get_column_name_mapping,
    load_dataset,
    process_data,
    save_dataset,
)

INFO_PATH = Path("data/Info")

# Dataset preprocessing registry
DATASET_REGISTRY = {}


def register_dataset(name):
    """Decorator to register a dataset preprocessing function."""

    def wrapper(func):
        DATASET_REGISTRY[name] = func
        return func

    return wrapper


# ---------------- Preprocessing functions ---------------- #


@register_dataset("housing")
def preprocess_housing(info):
    data_df = pd.read_csv(info["data_path"])
    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(info["data_path"], index=False)
    return df_cleaned


@register_dataset("yolanda_ml")
def preprocess_yolanda(info):
    columns = [f"feat_{idx}" for idx in range(100)]
    data_df = pd.read_csv(
        Path(info["data_path"]).with_suffix(".data"),
        sep=" ",
        names=columns,
        header=None,
        index_col=False,
    )
    data_df.to_csv(Path(info["data_path"]).with_suffix(".csv"), index=False)
    return data_df


@register_dataset("beijing")
def preprocess_beijing(info):
    data_df = pd.read_csv(info["raw_data_path"])
    data_df = data_df.iloc[:, 1:]  # drop first column
    df_cleaned = data_df.dropna()
    df_cleaned.to_csv(info["data_path"], index=False)
    return df_cleaned


@register_dataset("beijing_dcr")
def preprocess_beijing_dcr(info):
    return preprocess_beijing(info)


def preprocess_news_generic(info, remove_cat=False):
    data_df = pd.read_csv(info["raw_data_path"]).drop("url", axis=1)
    columns = np.array(data_df.columns.tolist())
    cat_columns1 = columns[12:18]
    cat_columns2 = columns[30:38]

    if not remove_cat:
        cat_col1 = data_df[cat_columns1].astype(int).to_numpy().argmax(axis=1)
        cat_col2 = data_df[cat_columns2].astype(int).to_numpy().argmax(axis=1)

    data_df = data_df.drop(cat_columns1.tolist() + cat_columns2.tolist(), axis=1)

    if not remove_cat:
        data_df["data_channel"] = cat_col1
        data_df["weekday"] = cat_col2

    data_save_path = f"data/{info['name']}/{info['name']}.csv"
    data_df.to_csv(data_save_path, index=False)
    info["data_path"] = data_save_path
    info["num_col_idx"] = list(range(45))
    info["cat_col_idx"] = [46, 47] if not remove_cat else []
    info["target_col_idx"] = [45]

    # Save updated info
    with open(f"{INFO_PATH}/{info['name']}.json", "w") as f:
        json.dump(info, f, indent=4)
    return data_df


@register_dataset("news")
def preprocess_news(info):
    info["name"] = "news"
    return preprocess_news_generic(info, remove_cat=False)


@register_dataset("news_nocat")
def preprocess_news_nocat(info):
    info["name"] = "news_nocat"
    return preprocess_news_generic(info, remove_cat=True)


@register_dataset("news_dcr")
def preprocess_news_dcr(info):
    info["name"] = "news_dcr"
    return preprocess_news_generic(info, remove_cat=False)


@register_dataset("news_nocat_dcr")
def preprocess_news_nocat_dcr(info):
    info["name"] = "news_nocat_dcr"
    return preprocess_news_generic(info, remove_cat=True)


def generic_loader(info, data_name):
    info["name"] = data_name
    return load_dataset(info)


@register_dataset("adult")
def preprocess_adult(info):
    generic_loader(info, "adult")


# ---------------- Core Processing ---------------- #


def load_info(name):
    with open(f"{INFO_PATH}/{name}.json", "r") as f:
        return json.load(f)


def get_dataset(
    config: omegaconf.DictConfig | omegaconf.OmegaConf, persist=False, seed: int = 1234
):
    dataset_name = config.name
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {dataset_name} is not registered.")

    info = load_info(dataset_name)
    DATASET_REGISTRY[dataset_name](info)
    data_df = load_dataset(info)
    train_df, test_df, val_df, info = process_data(
        dataset_name,
        info,
        data_df,
        seed=seed,
    )

    if persist:
        column_names = info.get("column_names") or data_df.columns.tolist()
        num_col_idx = info["num_col_idx"]
        cat_col_idx = info["cat_col_idx"]
        target_col_idx = info["target_col_idx"]
        num_columns = [column_names[i] for i in num_col_idx]
        cat_columns = [column_names[i] for i in cat_col_idx]
        target_columns = [column_names[i] for i in target_col_idx]

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

        has_val = bool(info.get("val_path"))
        save_dataset(
            train_df,
            test_df,
            val_df,
            num_columns,
            cat_columns,
            target_columns,
            dataset_name,
            has_val,
        )
    return train_df, test_df, val_df, info
