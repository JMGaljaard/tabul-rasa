import glob
import hydra
import json
import logging
import numpy as np
import omegaconf
import os
import pickle
import torch
import wandb
from pathlib import Path
from tabdiff.models import Trainer, UnifiedCtimeDiffusion
from tabdiff.models.modules import TabDiffDenoiseFn, UniModMLP
from tabdiff.utils.shared import get_dataloaders, get_metadata_dataset, get_metrics, get_save_path


@hydra.main(config_path="configs", config_name="tabdiff.yaml", version_base="1.6")
def main(cfg: omegaconf.DictConfig):
    device = "cuda"

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

    ckpt_parent_path = Path(cfg.model_save_path)

    ckpt_path_arr = list(ckpt_parent_path.glob("best_ema_model*"))
    assert (
        ckpt_path_arr
    ), f"Cannot not infer ckpt_path from {ckpt_parent_path}, please make sure that you first train a model before testing!"
    ckpt_path = ckpt_path_arr[0]

    ## Creat model_save and result paths
    model_save_path, result_save_path = get_save_path(cfg)

    ## Load training data
    categories, d_numerical, train_data, train_loader, val_data, val_loader = (
        get_dataloaders(cfg)
    )

    ## Load Metrics
    metrics = get_metrics(cfg, dataname, device, info)

    ## Load the module and models
    with omegaconf.open_dict(cfg):
        logging.warning("Updating the config object")
        cfg.diffusion.unimodmlp_params.d_numerical = d_numerical
        cfg.diffusion.unimodmlp_params.categories = (categories + 1).tolist()

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
        y_only_model_path = Path(cfg.y_only_model_pathi)

        y_only_model_path_arr = y_only_model_path.glob("best_ema_model*")
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

    diffusion = UnifiedCtimeDiffusion.from_config(
        cfg.diffusion, denoise_fn=denoise_fn, y_only_model=y_only_model
    )
    num_params = sum(p.numel() for p in diffusion.parameters())
    print("The number of parameters = ", num_params)
    diffusion.to(device)
    diffusion.train()

    ## Print the configs
    printed_configs = omegaconf.OmegaConf.to_yaml(cfg, resolve=True)
    print(f"The config of the current run is : \n {printed_configs}")

    # ## Enable Wandb
    project_name = f"tabdiff_{dataname}"
    with omegaconf.open_dict(cfg):
        cfg.project_name = project_name
    logger = wandb.init(
        project=cfg.project_name,
        name=exp_name,
        config=dict(cfg),
        mode="disabled",
    )

    ## Load Trainer
    sample_batch_size = cfg.sample.batch_size
    trainer = Trainer(
        diffusion,
        train_loader,
        train_data,
        val_data,
        val_loader,
        metrics,
        logger,
        **cfg.train.main,
        sample_batch_size=sample_batch_size,
        model_save_path=cfg.model_save_path,
        result_save_path=cfg.result_save_path,
        device=device,
        ckpt_path=ckpt_path,
        y_only=cfg.experiment.name == "imputation",
    )

    match cfg.experiment.name.lower():
        case "dcr":
            trainer.report_test_dcr(num_runs=cfg.epxeriment.num_runs)
        case "quality":
            trainer.report_test(num_runs=cfg.experiment.num_runs)
        case "impute":
            imputed_sample_save_dir = Path(f"impute/{dataname}/{exp_name}")
            trainer.test_impute(
                cfg.experiment.trial_start,
                cfg.experiment.trial_size,
                cfg.experiment.resample_rounds,
                cfg.experiment.impute_condition,
                imputed_sample_save_dir,
                cfg.experiment.w_num,
                cfg.experiment.w_cat,
            )

    ## Save config
    config_save_path = cfg.model_save_path
    with open(os.path.join(config_save_path, "config.pkl"), "wb") as f:
        pickle.dump(dict(cfg), f)
    with open(os.path.join(config_save_path, "tester.json"), "w") as f:
        import yaml

        json.dump(yaml.safe_load(omegaconf.OmegaConf.to_yaml(cfg, resolve=True)), f)


def prepare_imputation_learner(cfg, dataname, exp_name, info, *args, **kwargs):
    main_model_path = cfg.ckpt_path
    if main_model_path is None:
        main_model_parent_path = Path(
            f"./ckpt/{dataname}/{exp_name.replace('_y_only', '')}"
        )
        main_model_path_arr = list(
            main_model_parent_path.glob(
                f"{re.escape(str(main_model_parent_path))}/best_ema_model*"
            )
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
                    num_numerical=cfg.diffusion.unimodmlp_params.d_numerical,
                    **dict(cfg.diffusion.noise_schedule_params),
                )
                cfg.diffusion_params.noise_schedule_params.rho = noise_schedule.rho()[
                    0
                ].item()  # the target col is placed at the first position
            else:
                noise_schedule = LogLinearNoise_PerColumn(
                    num_categories=len(cfg.diffusion.unimodmlp_params.categories),
                    **main_model_configs["diffusion_params"]["noise_schedule_params"],
                )
                cfg.diffusion_params.noise_schedule_params.k = noise_schedule.k()[
                    0
                ].item()  # the target col is placed at the first position


if __name__ == "__main__":
    main()
