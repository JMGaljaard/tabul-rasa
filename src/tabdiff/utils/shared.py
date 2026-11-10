import json
import os
from pathlib import Path
import random
from typing import Optional, Tuple

import numpy as np
import omegaconf
import torch
from sklearn.base import TransformerMixin
from torch.utils.data import DataLoader, default_collate

import wandb
from tabdiff.datasets.datasets import TabDiffDataset
from tabdiff.datasets.transforms import BaseTensorTransform
from tabdiff.eval import TabMetrics
from tabdiff.unlearn.eval import TabULMetrics, TabNLLMetrics
from .data import Dequantizer, remove_metadata_column


def get_save_path(cfg: omegaconf.DictConfig) -> str:
    """

    Args:
        cfg (omegaconf.DictConfig): Retrieve document to write to, fakka

    Returns:
        str: Path pointing to the save location, derived from the training objective.
    """
    model_save_path, result_save_path = Path(cfg.model_save_path), Path(
        cfg.result_save_path
    )
    if True:
        # DO train
        model_save_path = "debug/ckpt" if cfg.experiment.debug else cfg.model_save_path
        result_save_path = (
            model_save_path.replace("ckpt", "result")
            if cfg.experiment.debug
            else cfg.result_save_path
        )
    elif args.mode == "test":
        #     TODO extract evaluation from the training loop...
        if args.report:
            result_save_path = f"eval/report_runs/{exp_name}/{dataname}"
        else:
            result_save_path = os.path.dirname(ckpt_path).replace(
                "ckpt", "result"
            )  # infer the exp_name from the ckpt_name
    for path in [model_save_path, result_save_path]:
        path = Path(path)
        if path is not None:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            else:
                print("Experiment was already run!")

    return model_save_path, result_save_path


def get_metadata_dataset(cfg):
    ## Get data info
    dataname = cfg.dataset.name
    info_path = f"data/{dataname}/info.json"
    with open(info_path, "r") as f:
        info = json.load(f)

    if hasattr(cfg.dataset, "preprocessing"):
        if cfg.dataset.preprocessing.method == "drop":
            print(cfg.dataset.preprocessing.columns)
            for indx in sorted(cfg.dataset.preprocessing.columns, reverse=True):
                info = remove_metadata_column(info, indx)
    return dataname, info


def get_dataloaders(
    cfg: omegaconf.DictConfig,
    maybe_fitted_transforms: Optional[
        Tuple[
            Optional[TransformerMixin],
            Optional[Dequantizer],
            Optional[TransformerMixin],
        ]
    ],
) -> (
    np.ndarray,
    np.ndarray,
    TabDiffDataset,
    DataLoader,
    TabDiffDataset,
    DataLoader,
):
    """
    Construct Dataset from experimental configuration file.
    Args:
        cfg (omegaconf.DictConfig): Experimental config.

    Returns:
        np.ndarray: Number of categorica values for discrete features.
        np.ndarray: Dimensionality of numerical features.
        TabDiffDataset: Trainnig dataset.
        DataLoader: Training dataset DataLoader.
        TabDiffDataset: Test dataset.
        DataLoader: Test dataset DataLoader.
    """
    batch_size = cfg.train.main.batch_size
    training_imputation = cfg.experiment.name == "imputation"

    train_data = TabDiffDataset.from_config(
        cfg,
        y_only=training_imputation,
        is_train=True,
        fitted_transforms=maybe_fitted_transforms,
    )
    d_numerical, categories = train_data.d_numerical, train_data.categories

    collate_fn = (
        BaseTensorTransform.from_config(
            cfg,
            collate_fn=default_collate,
        )
        or default_collate
    )
    # TODO: Indicate how to split the data. Currenlty this is a bit messed up if...
    #   Do we want to have a split of data?
    #   If so, how do
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=5,
        prefetch_factor=40,
        pin_memory=True,
        pin_memory_device="cuda",
        collate_fn=collate_fn,
        persistent_workers=True,
    )
    val_data = TabDiffDataset.from_config(
        cfg.dataset,
        y_only=training_imputation,
        is_train=False,
        fitted_transforms=maybe_fitted_transforms,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=30_000,
        shuffle=False,
        num_workers=5,
        prefetch_factor=40,
        persistent_workers=True,
    )
    return categories, d_numerical, train_data, train_loader, val_data, val_loader


def get_metrics(cfg, dataname, device, info):
    real_data_path, test_data_path, val_data_path = get_experiment_data_paths(dataname)
    if not os.path.exists(val_data_path):
        print(
            f"{cfg.dataset.name} does not have its validation set. During MLE evaluation, a validation set will be splitted from the training set!"
        )
        val_data_path = None
    # if args.mode == "train":

    match cfg.experiment.name:
        case "train":
            if hasattr(cfg, "unlearn") and cfg.unlearn.name == "feature":
                if hasattr(cfg.unlearn.main, "metric_list"):
                    metric_list = cfg.unlearn.main.metric_list
                else:
                    metric_list = ["density", "mle", "c2st", "correlational_ul"]
                # Make use of evaluation instead?
                ul_config = cfg.unlearn

                if not "ddpm" in cfg.diffusion.exp_name_model:
                    metrics = TabNLLMetrics.from_config(
                        config=cfg,
                        real_data_path=real_data_path,
                        test_data_path=test_data_path,
                        val_data_path=val_data_path,
                        info=info,
                        device=device,
                        reference_model=...,
                        unlearn_model=...,
                    )
                else:
                    metrics = TabULMetrics(
                        ul_config,
                        real_data_path,
                        test_data_path,
                        val_data_path,
                        info,
                        device,
                        dataset_conf=cfg.dataset,
                        metric_list=metric_list,
                    )
                return metrics
            metric_list = ["density"]
        case "dcr":
            metric_list = ["dcr"]
        case _:
            if hasattr(cfg.train, "metric_list"):
                metric_list = cfg.train.metric_list
            else:
                metric_list = [
                    "density",
                    "mle",
                    "c2st",
                ]

    metrics = TabMetrics(
        real_data_path,
        test_data_path,
        val_data_path,
        info,
        device,
        metric_list=metric_list,
        config=cfg,
    )
    return metrics


def get_experiment_data_paths(dataname: str) -> Tuple[str, str, str]:
    real_data_path = f"synthetic/{dataname}/real.csv"
    test_data_path = f"synthetic/{dataname}/test.csv"
    val_data_path = f"synthetic/{dataname}/val.csv"
    return real_data_path, test_data_path, val_data_path


def seed_all(cfg):
    # TODO: Use torch.lighngin for this
    print("DETERMINISTIC MODE is enabled!!!")
    ## Set global random seeds
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    ## Ensure deterministic CUDA operations
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ':16:8'
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_wandb(cfg: omegaconf.DictConfig):
    wandb.config = wandb_conf = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    logger = wandb.init(
        # project=cfg.project_name,
        # name="test-run-{}",
        # mode="disabled",
        name=cfg.wandb.name,  # https://github.com/wandb/wandb/issues/1192#issuecomment-2344715927 Don't use ID.
        group=cfg.wandb.group,
        job_type=cfg.wandb.job_type,
        config=wandb_conf,
        reinit=True,
    )
    return logger
