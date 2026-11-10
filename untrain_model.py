import dotenv
import glob
import hydra
import json
import logging
import numpy as np
import omegaconf
import os
import pickle
import pickle as pkl
import re
import torch
import traceback
import wandb
from omegaconf import OmegaConf
from pathlib import Path
from sklearn.base import TransformerMixin
from tabdiff.models import UnifiedCtimeDiffusion, UnifiedDtimeDiffusion
from tabdiff.models.modules import TabDiffDenoiseFn, UniModMLP
from tabdiff.unlearn.callback import UnlearnCallback
from tabdiff.unlearn.eval import TabULMetrics
from tabdiff.unlearn.trainer import UnTrainer
from tabdiff.utils import Dequantizer
from tabdiff.utils.shared import (
    get_dataloaders,
    get_experiment_data_paths,
    get_metadata_dataset,
    get_save_path,
    init_wandb,
    seed_all,
)
from typing import Optional, Tuple


def get_latest_train_if_exists(cfg, prefix: str = "best_model_",) -> Optional[
    Tuple[
        Path,
        Optional[
            Tuple[
                Optional[TransformerMixin],
                Optional[Dequantizer],
                Optional[TransformerMixin],
            ]
        ],
    ]
]:
    load_directory = Path(cfg.model_load_path)
    # Get all checkpoint files in the directory that match the pattern
    checkpoint_files = list(load_directory.glob(f"{prefix}*.pt"))
    data_transform_filepath = load_directory / "transforms.pkl"

    def get_last_checkpoint(dir_path):
        # Convert the directory path to a Path object
        if not checkpoint_files:
            return None  # No checkpoint files found

        # Define a function to extract a sortable index
        def extract_sort_key(file_name):
            # Match a pattern like 'checkpoint_1.pt', 'checkpoint_epoch_10.pt', or 'checkpoint_epoch_final.pt'
            match = re.search(r"checkpoint_(\d+|final|\D+)", file_name.stem)
            if match:
                # Return a tuple: (is_number, the extracted value)
                value = match.group(1)
                return (value.isdigit(), int(value) if value.isdigit() else value)
            return (False, file_name.stem)  # Fallback for unmatched patterns

        # Sort the files based on the extracted value: first by whether it's numeric or not, then by the value itself
        checkpoint_files.sort(key=extract_sort_key, reverse=True)

        # Return the most recent checkpoint file
        return checkpoint_files[0]

    transformation_dict = None
    try:
        with open(data_transform_filepath, "rb") as f:
            pkl.load(f)
    except Exception as e:
        logging.warning(
            f"Loaded model does not provide expected encodings for dataset!"
        )

    if checkpoint_files:
        return get_last_checkpoint(load_directory), transformation_dict


def prepare_callbacks(
    cfg: omegaconf.DictConfig,
    categories: np.ndarray = None,
) -> UnlearnCallback:
    """
    Helper function to construct callback functions for unlearning, i.e., create unlearning objective.
    Args:
        cfg (omageconf.DictConfig): Experiment configuration.
        categories (np.ndarray): Array of cardinality of each categorical features, includes the 'mask' token.

    Returns:
        UnlearnCallback: Configured Unlearning callback function.
    """
    unlearn_cfg = cfg.unlearn
    num_callback, cat_callback = None, None
    # TODO: Ensure we have a
    match unlearn_cfg.orientation:
        case "vertical":

            # Case of feature(s) unlearning

            if hasattr(unlearn_cfg, "column") and unlearn_cfg.column is not None:
                # TODO: figure out multi-column unlearning
                dtype = unlearn_cfg.dtype
                # Check if valid
                # Problem, we assume only updates are required by a single colum
                # we want logic that alters the behavior conditionally on multiple dtypes
                # However. this no workie.

                # TODO: address multiple columns..
                # HAave a type 'mixed' or drop it altogether...

                if dtype == "numerical":
                    # TabDiff will use first part as numerical
                    cat_callback = UnlearnCallback.from_config(
                        cfg,
                        dtype="categorical",
                        categories=categories,
                    )
                    num_callback = UnlearnCallback.from_config(
                        cfg,
                        dtype="numerical",
                    )
                else:
                    cat_callback = UnlearnCallback.from_config(
                        cfg,
                        dtype="categorical",
                        categories=categories,
                    )
                    num_callback = UnlearnCallback.from_config(
                        cfg,
                        dtype="numerical",
                    )

        case "horizontal":
            raise NotImplementedError("Horizonal unlearning not yet supported")

    return num_callback, cat_callback


@hydra.main(
    config_path="configs", config_name="tabdiff_unlearn_sweep.yaml", version_base="1.6"
)
def main(cfg: omegaconf.DictConfig):
    # UGHHGHGHG
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    #
    device = "cuda"

    dotenv.load_dotenv()

    try:
        current_num = hydra.core.hydra_config.HydraConfig.get().job.num
    except omegaconf.errors.MissingMandatoryValue:
        print("Running without sweep!")
        current_num = 0
    local_rank = int(os.getenv("SLURM_NODEID", 0))
    parallel_tasks = int(os.getenv("SLURM_NNODES", 1))
    if (current_num % parallel_tasks) != local_rank:
        # In this case, we are bounded to sod-off
        logging.info(
            f"Skipping task of id: {current_num}, local_id: {local_rank}, context: {parallel_tasks}"
        )
        return

    ## Disable scientific numerical format
    np.set_printoptions(suppress=True)
    torch.set_printoptions(sci_mode=False)

    dataname, info = get_metadata_dataset(cfg)

    ## Set up flags
    # TODO: convert to config.experiment.nam
    is_dcr = "dcr" in cfg.experiment.name

    ## Set experiment name
    exp_name = cfg.exp_name
    if cfg.exp_name is None:
        exp_name = cfg.diffusion.exp_name_model
    # TODO: Fix me.
    # exp_name += "_y_only" if args.y_only else ""

    # TODO: restart training form checkpoitn not working as these smarty pants forgor the optimizer.\
    try:

        ckpt_path, maybe_transform_dict = get_latest_train_if_exists(
            cfg, prefix="best_model_"
        )
    except:
        raise ValueError("Could not find moel")
    print("Unlearning is started")

    ## Creat model_save and result paths
    model_save_path, result_save_path = get_save_path(cfg)
    if cfg.experiment.deterministic:
        seed_all(cfg)

    # ## Set debug mode parameters
    if cfg.experiment.debug:  # fast eval for DEBUG mode
        with omegaconf.open_dict(cfg):
            cfg.train.main.check_val_every = 2
            cfg.train.main.batch_size = 4096
            cfg.sample.batch_size = 10000
            cfg.diffusion.num_timesteps = 4

    # TODO: Ensure data is loaded from a configuration
    #   Acct. for training data
    #   Acct. for testing data
    #   Acct. for valididation data
    ## Load training data
    fitted_transforms = None

    # TODO: Get this act together.k
    if isinstance(maybe_transform_dict, dict):
        num_transform = maybe_transform_dict.get("num_transform", None)
        int_transform = maybe_transform_dict.get("int_transform", None)
        cat_transform = maybe_transform_dict.get("cat_transform", None)

        fitted_transforms = (num_transform, int_transform, cat_transform)

    categories, d_numerical, train_data, train_loader, val_data, val_loader = (
        get_dataloaders(cfg, maybe_fitted_transforms=fitted_transforms)
    )

    ## Load the module and models
    with omegaconf.open_dict(cfg):
        logging.warning("Updating the config object")
        cfg.diffusion.unimodmlp_params.d_numerical = d_numerical
        cfg.diffusion.unimodmlp_params.categories = (categories + 1).tolist()

    # Load model
    diffusion = diffusion_from_config(cfg, dataname, device, exp_name, info)
    num_params = sum(p.numel() for p in diffusion.parameters())
    print(f"The number of parameters = {num_params}", flush=True)
    diffusion.to(device)
    diffusion.train()
    ## Load Metrics
    real_data_path, test_data_path, val_data_path = get_experiment_data_paths(
        dataname=dataname,
    )

    # TODO: Efficient evaluatoin
    #   1. Append / include in dataframe
    #   2. Parallel dispatch of evaluation
    #   3. Delayed computation???
    if hasattr(cfg.unlearn.main, "metric_list"):
        evaluation_metrics = cfg.unlearn.main.metric_list
    else:
        evaluation_metrics = ["density", "mle", "c2st", "correlational_ul"]
    print(evaluation_metrics)
    metrics = TabULMetrics(  # .from_config(
        # cfg,
        # evaluation_metrics=evaluation_metrics,
        metric_list=evaluation_metrics,
        real_data_path=real_data_path,
        test_data_path=test_data_path,
        val_data_path=val_data_path,
        info=info,
        device=device,
        config=cfg,
        # reference_model=None,
        # unlearn_model=diffusion,
    )

    # ## Enable Wandb
    # project_name = f"tabdiff_{dataname}"
    # with omegaconf.open_dict(cfg):
    #     cfg.project_name = project_name

    logger = init_wandb(cfg)

    ## Load Trainer
    # Initialize the callback for unlearning (inject loss hooks)
    unlearn_callback_num, unlearn_callback_cat = prepare_callbacks(
        cfg,
        categories=categories,
    )

    sample_batch_size = cfg.sample.batch_size
    trainer = UnTrainer.from_config(
        cfg,
        diffusion,
        train_loader,
        train_data,
        val_data,
        val_loader,
        metrics,
        logger,
        update_fn_num=unlearn_callback_num,
        update_fn_cat=unlearn_callback_cat,
        # Overwrite the defaults using unlearning parameters
        **cfg.unlearn.main,
        sample_batch_size=sample_batch_size,
        model_save_path=cfg.model_save_path,
        result_save_path=cfg.result_save_path,
        device=device,
        ckpt_path=ckpt_path,
        y_only=cfg.experiment.name == "imputation",
    )

    try:
        trainer.run_loop()
        config_save_path = Path(cfg.unlearn.model_save_path)
        if not config_save_path.is_dir() and not config_save_path.exists():
            config_save_path.mkdir(parents=True, exist_ok=True)

        with open(os.path.join(config_save_path, "config.pkl"), "wb") as f:
            pickle.dump(dict(cfg), f)
        with open(os.path.join(config_save_path, "tester.json"), "w") as f:
            import yaml

            json.dump(yaml.safe_load(omegaconf.OmegaConf.to_yaml(cfg, resolve=True)), f)
        trainer.run_loop()
    except Exception as e:
        traceback.print_exc()
        wandb.finish()


def diffusion_from_config(cfg, dataname, device, exp_name, info):
    import torch

    # TODO: Make evaluation / tasks seperate configs / seperate files to execute / provide setup.
    if cfg.experiment.name == "imputation":
        prepare_imputation_learner(cfg, dataname, exp_name, info)
    backbone = UniModMLP.from_config(cfg.diffusion.unimodmlp_params)
    denoise_fn = TabDiffDenoiseFn.from_config(
        cfg.diffusion.edm_params, denoise_fn=backbone
    )
    denoise_fn.to(device)
    ## Create and load y_only_model for imputation
    y_only_model = None
    if cfg.experiment.name == "imputation":
        y_only_model_path = cfg.y_only_model_path

        y_only_model_path_arr = glob.glob(f"{y_only_model_path}/best_ema_model*")
        assert (
            y_only_model_path_arr
        ), f"Cannot not infer y_only model's ckpt_path from {y_only_model_path}, please make sure that you first train a y_only model before testing imputation!"
        y_only_model_path = y_only_model_path_arr[0]
        y_only_model_config_path = os.path.join(
            os.path.dirname(y_only_model_path), "config.yaml"
        )

        y_only_model_config = omegaconf.OmegaConf.load(y_only_model_config_path)
        y_only_model = UniModMLP.from_config(
            y_only_model_config.diffusion.unimodmlp_params,
        )
        y_only_model = TabDiffDenoiseFn.from_config(
            cfg.diffusion.edm_params, denoise_fn=y_only_model
        )
        y_only_model.to(device)
        # load weights
        state_dicts = torch.load(y_only_model_path, map_location=device)
        y_only_model.load_state_dict(state_dicts["denoise_fn"])
    if "ddpm" in cfg.diffusion.exp_name_model:
        diffusion = UnifiedDtimeDiffusion.from_config(
            cfg.diffusion,
            denoise_fn=denoise_fn,
            y_only_model=y_only_model,
        )
    else:
        if "edm" in cfg.diffusion.exp_name_model:
            diffusion = UnifiedDtimeDiffusion.from_config(
                cfg.diffusion,
                denoise_fn=denoise_fn,
                y_only_model=y_only_model,
            )
        else:
            diffusion = UnifiedCtimeDiffusion.from_config(
                cfg.diffusion,
                denoise_fn=denoise_fn,
                y_only_model=y_only_model,
            )
    return diffusion


def prepare_imputation_learner(cfg, dataname, exp_name, info):
    main_model_path = cfg.ckpt_path
    if main_model_path is None:
        main_model_parent_path = Path(
            f"./ckpt/{dataname}/{exp_name.replace('_y_only', '')}"
        )
        main_model_path_arr = list(
            main_model_parent_path.glob(f"{main_model_parent_path}/best_ema_model*")
        )
        assert (
            main_model_path_arr
        ), f"Cannot not infer the main model's ckpt_path from {main_model_parent_path}, please make sure that you first train a main model before training the y_only model!"
        main_model_path = main_model_path_arr[0]
    main_model_configs = pickle.load(
        open(os.path.join(os.path.dirname(main_model_path), "config.pkl"), "rb")
    )
    if (
        cfg.diffusion.ex_name_model == "learnable_schedule"
    ):  # if learnable schedule is enabled in the main model, we need to infer noise params of the target column from the main model ckpt and train the y_only model with those params
        from tabdiff.models.noise_schedule import (
            PowerMeanNoise_PerColumn,
            LogLinearNoise_PerColumn,
        )

        with omegaconf.open_dict(cfg):
            # Update the config using the pre-trained model's scheduler as starting point.
            if info["task_type"] == "regression":
                noise_schedule = PowerMeanNoise_PerColumn(
                    num_numerical=main_model_configs["unimodmlp_params"]["d_numerical"],
                    **main_model_configs["diffusion_params"]["noise_schedule_params"],
                )
                raw_config["diffusion_params"]["noise_schedule_params"][
                    "rho"
                ] = noise_schedule.rho()[
                    0
                ].item()  # the target col is placed at the first position
            else:
                noise_schedule = LogLinearNoise_PerColumn(
                    num_categories=len(
                        main_model_configs["unimodmlp_params"]["categories"]
                    ),
                    **main_model_configs["diffusion_params"]["noise_schedule_params"],
                )
                raw_config["diffusion_params"]["noise_schedule_params"][
                    "k"
                ] = noise_schedule.k()[
                    0
                ].item()  # the target col is placed at the first position


if __name__ == "__main__":

    OmegaConf.register_new_resolver(
        "slice",
        lambda s, a, b: s[int(a) : min(len(s), int(b))],
    )
    main()
