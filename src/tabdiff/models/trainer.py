import abc
from collections import defaultdict
import glob
import json
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Queue
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import omegaconf
import pandas as pd
import scipy
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from tabdiff.datasets import TabDiffDataset
from tabdiff.datasets.processors import inverse_data_transform

from fastparquet import write
from tabdiff.datasets import TabDiffDataset
from tabdiff.datasets.processors.postprocessors import (
    inverse_data_transform,
    recover_data,
    split_num_cat_target,
)
from tabdiff.eval import TabMetrics
from tabdiff.models.auto import TrainerRegistry
from tabdiff.models.schedulers import AnnealSchedule
from tabdiff.models.schedulers.loss_schedulers import ConstantSchedule, ScaleSchedule
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.models.unified_ctime_diffusion import (
    MixedDiffusionModelOutput,
)

BAR = "=============="


def logging_init():
    logging.disable(logging.CRITICAL)
    import warnings

    # Suppress all warnings
    warnings.filterwarnings("ignore")

    # OR: suppress only sklearn convergence warnings
    try:
        from sklearn.exceptions import ConvergenceWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except ImportError:
        pass


def _evaluate_metric_worker(args) -> (str, dict, dict, int):
    """
    Top-level worker function (must be picklable).
    """
    # metric, evaluator, syn_data, train_set, test_set, validation_set = args
    metric, evaluator, syn_data = args
    print(f"Evaluating {metric}")
    func = getattr(evaluator, f"evaluate_{metric}")
    # Assume 0 status code fo
    status = 0
    out_metrics, out_extras = {}, {}
    try:
        out_metrics, out_extras = func(
            syn_data,
            # train_set=train_set,
            # test_set=test_set,
            # validation_set=validation_set,
        )
    except Exception:
        status = 1
        print("Evaluation failed, likely due to mode collapse!")
        print(traceback.format_exc())

    return metric, out_metrics, out_extras, status


def print_with_bar(log_msg):
    log_msg = BAR + log_msg + BAR
    if "End" in log_msg:
        log_msg += "\n"
    print(log_msg)


def unpack_mixed_loss(
    mixed_output: MixedDiffusionModelOutput,
    mask_index: torch.Tensor,
    reduce: bool = True,
    scale: bool = True,
    device: torch.device = torch.device("cpu"),
    **kwargs,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """
    Unpack and Scale a MixedDiffusionModelOutput to instances of torch.FloatTensor.

    Notes:
        This function safely unpacks and scales (if available) a MixedDiffusionOutput.
    Args:
        mixed_output (MixedDiffusionModelOutput): MixedDiffusionModelOutput with optional fields to unpack.
        mask_index (torch.Tensor): Indices representing the mask index of the model, i.e., to ignore activations
            in loss.
        reduce (bool, *, True): By default reduce the output using torch.mean()
        device (torch.device): Device to use to store default tensor (cpu by default).
    Returns:
        torch.FloatTensor: Continuous loss.
        torch.FloatTensor: Discrete loss.
    """
    # TODO: Check that we expect to only access weights here...
    cat_out = mixed_output.categorical_output
    con_out = mixed_output.continuous_output

    # Safely unpack the tensors
    dloss, closs = (
        (
            cat_out.loss
            if cat_out is not None
            else torch.zeros((1,), dtype=torch.float32, device=device)
        ),
        (
            con_out.loss
            if con_out is not None
            else torch.zeros((1,), dtype=torch.float32, device=device)
        ),
    )

    if cat_out is not None and mask_index is not None:
        # NOTE: This is different from NPO2, where we simply take the average?
        #   We assumed that ...
        # If catout defined, we will filter out 'unwanted'
        cat_input_masked_mask = (
            mixed_output.categorical_output.xt == mask_index
        ).float()
        # Only 'masked' positions in the input will have a gradient flow back. Others are zero-d out
        dloss = dloss * cat_input_masked_mask
    # Safely perform scaling with time component.
    #   Note, that we don't check for
    if scale:
        if closs is not None and (time_scale_con := con_out.weight) is not None:
            closs = closs * time_scale_con
        if dloss is not None and (time_scale_cat := cat_out.weight) is not None:
            dloss = dloss * time_scale_cat

    if reduce:
        closs = closs.mean()
        dloss = dloss.mean()
    return closs, dloss


class DiffusionTrainer(metaclass=TrainerRegistry):

    @abc.abstractmethod
    def from_config_(self, config: omegaconf.DictConfig, *args, **kwargs):
        raise NotImplementedError("Call specific implementation, not base")

    @classmethod
    def from_config(cls, config, *args, **kwargs):
        # Default to trainer
        if hasattr(config, "unlearn"):
            if (
                hasattr(config.unlearn, "trainer")
                and config.unlearn.trainer is not None
            ):
                return (
                    cls.get_registry()
                    .get(config.unlearn.trainer)
                    .from_config_(config, *args, **kwargs)
                )
            else:
                # If unlearning defined, we load an untrainer
                return cls.get_registry().get("UnTrainer")(config, *args, **kwargs)
        else:
            # By default load 'trainer'
            return cls.get_registry().get("Trainer")(*args, **kwargs)


def init_from_checkpoint(
    diffusion: UnifiedCtimeDiffusion, ckpt_path=None, device=torch.device("cpu")
):
    if ckpt_path is not None:
        logging.info(f"Loading pre-trained weights from checkpoint: {ckpt_path}")
        # TODO: Fix EMA models as well.
        state_dicts = torch.load(ckpt_path, map_location=device)
        prefix = "denoise_fn_D." if "ema" in str(ckpt_path) else ""
        diffusion._denoise_fn.load_state_dict(state_dicts[f"denoise_fn"])
        diffusion.num_schedule.load_state_dict(state_dicts["num_schedule"])
        diffusion.cat_schedule.load_state_dict(state_dicts["cat_schedule"])


def ensure_exists(save_path: str | Path) -> Path:
    """
    Ensure that requested path exists after calling.
    Args:
        save_path (): Path to create.

    Returns:
        Path: Path object to the requested location.

    """
    if not isinstance(save_path, Path):
        save_path = Path(save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    return save_path


class Trainer(DiffusionTrainer):

    @torch.no_grad()
    def evaluate_generation(
        self,
        syn_df: pd.DataFrame,
        save_metric_details=False,
        plot_density=False,
        ema=False,
    ):
        """
        Overwrite default evaluate behavior, as we are currenlty not interested in EMA behavior, we can skip the eval,
        and keep default behavior in-line with original repo.
        Args:
            epoch ():
            log_dict ():

        Returns:

        """
        self.diffusion.eval()

        # TODO: Move saving of data to ...
        # Sample a synthetic table
        num_samples = self.metrics.real_data_size

        save_dir = Path(self.result_save_path) / str(self.curr_epoch)
        if ema:
            save_dir = save_dir / "ema"
        save_dir.mkdir(exist_ok=True, parents=True)
        syn_internal = syn_df.copy()

        floating_cols = syn_df.select_dtypes(include="floating").columns
        syn_internal[floating_cols] = syn_internal[floating_cols].astype("float64")


        tasks = [
            (
                metric,
                self.metrics,  # evaluator instance
                syn_internal.copy(),
                # train_ds,
                # test_ds,
                # None,
            )
            for metric in self.metrics.metric_list
        ]
        metrics, extras = dict(), dict()

        with ProcessPoolExecutor(
            max_workers=self._config.experiment.evaluation_parallelism,
            initializer=logging_init,
        ) as executor:

            # TODO: Unroll all requested executions by the exector pool...
            print("Running Stat tasks")
            futures = [
                executor.submit(_evaluate_metric_worker, task)
                for task in tasks
                if "mle" not in task[0]
            ]
            print("Running MLE tasks")
            mle_tasks = [task for task in tasks if "mle" in task[0]]
            if len(mle_tasks) > 0:
                # Perform MLE seperately.
                (
                    metric,
                    metrics_exectutor,  # evaluator instance
                    syn_internal_copy,
                    # ds,
                    # test_ds,
                    # val_ds,
                ) = mle_tasks[0]

                out_metrics, out_extras = {}, {}
                try:
                    # TODO: Make it possible to make augment not shift...
                    out_metrics, out_extras = metrics_exectutor.evaluate_mle(
                        syn_internal_copy,
                        executor=executor,
                        # train_set=train_ds,
                        # test_set=test_ds,
                        # validation_set=val_ds,
                    )
                except Exception:
                    print("Evaluation failed, likely due to mode collapse!")
                    print(traceback.format_exc())
                else:
                    metrics.update(out_metrics)
                    extras.update(out_extras)

            for future in as_completed(futures):
                metric, out_metrics, out_extras, status = future.result()
                print(f"{metric} completed with status code {status}")
                metrics.update(out_metrics)
                extras.update(out_extras)

        # Save metrics and metric details
        result_path = save_dir / "all_results.json"
        with open(result_path, "w") as json_file:
            json.dump(
                metrics, json_file, indent=4, separators=(", ", ": ")
            )  # always locally save the output metrics

        if save_metric_details:
            for name, extra in extras.items():

                if isinstance(extra, pd.DataFrame):
                    detail_path_csv = save_dir / f"{name}.csv"
                    extra.to_csv(detail_path_csv)
                elif isinstance(extra, dict):
                    detail_path_json = save_dir / f"{name}.json"
                    with open(detail_path_json, "w") as json_file:
                        json.dump(extra, json_file, indent=4, separators=(", ", ": "))
                else:
                    raise NotImplementedError(
                        f"Extra file generated during evaluations has type {type(extra)}, and code to save this type of file is not implemented"
                    )

        # Plot density figures
        if plot_density:
            img = self.metrics.plot_density(syn_df)
            density_plot_path = save_dir / "density_plots.pdf"
            img.save(density_plot_path)
        return metrics, extras, syn_df

    def from_config_(self, config, *args, **kwargs):
        # For default reasons, we also provide backwards compatibility by defaulting ot paper values
        steps = (
            config.main.steps
            if hasattr(config, "train")
            and hasattr(config.train, "main")
            and hasattr(config.train, "steps")
            else 8000
        )

        closs_scale_or_default = AnnealSchedule(loss_lambda=1.0, total_steps=steps)
        dloss_scale_or_default = ConstantSchedule(loss_lambda=1.0)

        try:
            closs_scale_or_default = ScaleSchedule.from_config(
                config.main.loss_schedules.discrete,
            )
        except Exception as e:
            self.logger.warning(e, exc_info=True)
            self.logger.warning(
                f"Defaulting to paper defaults continuout: steps={steps} scale=1.0"
            )
        try:
            dloss_scale_or_default = ScaleSchedule.from_config(
                config.main.loss_schedules.continuous,
            )
        except Exception as e:
            self.logger.warning(e, exc_info=True)
            self.logger.warning(
                f"Defaulting to paper defaults continuout: steps={steps} scale=1.0"
            )

        evaluation_parallelism = (
            10
            if hasattr(config.experiment, "evaluation_parallelism")
            else config.experiment.evaluation_parallelism
        )
        return Trainer(
            config,
            *args,
            closs_schedule=closs_scale_or_default,
            dloss_schedule=dloss_scale_or_default,
            evaluation_parallelism=evaluation_parallelism,
            **kwargs,
        )

    def __init__(
        self,
        config,
        diffusion,
        train_iter,
        dataset: TabDiffDataset,
        test_dataset,
        test_iter,
        metrics: TabMetrics,
        logger,
        lr,
        weight_decay,
        steps,
        batch_size,
        check_val_every,
        sample_batch_size,
        model_save_path,
        result_save_path,
        lr_scheduler="reduce_lr_on_plateau",
        reduce_lr_patience=100,
        factor=0.9,
        ema_decay=0.997,
        closs_schedule: ScaleSchedule = None,
        dloss_schedule: ScaleSchedule = None,
        device=torch.device("cuda:0"),
        ckpt_path=None,
        y_only=False,
        scale_numerical: bool = False,
        scale_categorical: bool = True,
        evaluation_parallelism: int = 10,
        **kwargs,
    ):
        self._config = config
        self.y_only = y_only
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()
        self.ema_num_schedule = deepcopy(self.diffusion.num_schedule)
        for param in self.ema_num_schedule.parameters():
            param.detach_()
        self.ema_cat_schedule = deepcopy(self.diffusion.cat_schedule)
        for param in self.ema_cat_schedule.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.test_iter = test_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(
            self.diffusion.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.ema_decay = ema_decay
        self.lr_scheduler = lr_scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=factor,
            patience=reduce_lr_patience,
            verbose=True,
        )
        self.closs_weight_schedule = closs_schedule
        self.dloss_weight_schedule = dloss_schedule

        self.batch_size = batch_size
        self.sample_batch_size = sample_batch_size
        self.metrics = metrics
        self.logger = logger
        self.check_val_every = check_val_every

        self.device = device
        self.model_save_path = model_save_path
        self.result_save_path = result_save_path
        self.ckpt_path = ckpt_path
        init_from_checkpoint(self.diffusion, self.ckpt_path, self.device)

        self.curr_epoch = (
            int(os.path.basename(self.ckpt_path).split("_")[-1].split(".")[0])
            if self.ckpt_path is not None and not hasattr(self, "ckpt_path")
            else 0
        )

        self.scale_numerical = scale_numerical
        self.scale_categorical = scale_categorical
        self.evaluation_parallelism = evaluation_parallelism

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, closs_weight, dloss_weight, epoch: int = -1):
        x = x.to(self.device, non_blocking=True)

        self.diffusion.train()

        self.optimizer.zero_grad()

        mixed_output: MixedDiffusionModelOutput = self.diffusion.mixed_loss(x)
        closs, dloss = unpack_mixed_loss(
            mixed_output,
            # Expicitly set the mask index to that of the model, as otherwise we degenerate the solution
            self.diffusion.mask_index,
            scale=True,
            reduce=True,
            device=self.device,
        )

        loss = dloss_weight * dloss + closs_weight * closs
        loss.backward()
        self.optimizer.step()

        return dloss.item(), closs.item()

    @torch.no_grad()
    def compute_loss(
        self, prefix: str = ""
    ) -> Dict[str, Dict[str, float]]:  # eval loss is not weighted
        """Compute the loss of entire dataset with current model. To prevent EMA noise.

        Args:
            prefix (str): Model prefix

        Returns:

        """
        PRECISION = 4
        # Prepare accumulator
        ret_dict = defaultdict(
            lambda: {"curr_dloss": 0.0, "curr_closs": 0.0, "curr_count": 0}
        )

        data_iter = self.train_iter
        self.diffusion.eval()

        for batch in data_iter:
            if isinstance(batch, torch.Tensor):
                batch = {"samples": batch}

            for key, key_batch in batch.items():
                x = key_batch.float().to(self.device, non_blocking=True)

                mini_batch_size = len(x)
                model_output = self.diffusion.mixed_loss(x)
                batch_closs, batch_dloss = unpack_mixed_loss(
                    model_output,
                    self.diffusion.mask_index,
                    reduce=True,
                    scale=True,
                    device=self.device,
                )
                ret_dict[key]["curr_dloss"] += batch_dloss.item() * mini_batch_size
                ret_dict[key]["curr_closs"] += batch_closs.item() * mini_batch_size
                ret_dict[key]["curr_count"] += mini_batch_size

        flat_return_dict = {}
        for key, key_accum in ret_dict.items():
            key_dict = {
                f"loss/{key}/{prefix}dloss": (
                    dloss := np.around(
                        key_accum["curr_dloss"] / key_accum["curr_count"], PRECISION
                    )
                ),
                f"loss/{key}/{prefix}closs": (
                    closs := np.around(
                        key_accum["curr_closs"] / key_accum["curr_count"], PRECISION
                    )
                ),
                f"loss/{key}/{prefix}total_loss": np.around(dloss + closs, PRECISION),
            }
            flat_return_dict.update(key_dict)
        return flat_return_dict

    def run_loop(self):
        patience = 0
        best_loss = np.inf
        best_ema_loss = np.inf
        best_val_loss = np.inf
        start_time = time.time()
        print_with_bar(f"Starting Trainin Loop, total number of epoch = {self.steps}")
        # Set up wandb's step metric
        self.logger.define_metric("epoch")
        self.logger.define_metric("*", step_metric="epoch")

        start_epoch = self.curr_epoch

        # Stable SWA (EMA equiv.), more stable than original implementation...
        ema_update = torch.optim.swa_utils.get_ema_multi_avg_fn(decay=self.ema_decay)
        inner_loop_progress = tqdm(
            self.train_iter,
            total=len(self.train_iter),
            leave=False,
            position=0,
            desc="Step",
        )

        if start_epoch > 0:
            print_with_bar(
                f"Resuming training from epoch {start_epoch}, with validation check every {self.check_val_every} epoches"
            )
        nll_loss = None
        for epoch in tqdm(
            range(start_epoch, self.steps),
            total=self.steps - start_epoch,
            desc="Epochs",
            leave=True,
            position=1,
        ):
            self.curr_epoch = epoch + 1
            # Set up inner_loop_progress
            inner_loop_progress.reset()
            # inner_loop_progress.set_description(f"Epoch {epoch+1}/{self.steps}")

            # Compute the loss weights
            closs_weight = self.closs_weight_schedule.step(epoch)
            dloss_weight = self.dloss_weight_schedule.step(epoch)

            # Training Step
            curr_dloss = 0.0
            curr_closs = 0.0
            curr_count = 0
            curr_lr = self.optimizer.param_groups[0]["lr"]
            for batch in self.train_iter:
                batch_dloss, batch_closs = self._run_step(
                    batch,
                    closs_weight,
                    dloss_weight,
                    epoch=epoch,
                )
                num_samples = (
                    len(batch)
                    if isinstance(batch, torch.Tensor)
                    else len(batch["sample"])
                )
                curr_dloss += batch_dloss * num_samples
                curr_closs += batch_closs * num_samples
                curr_count += num_samples

                inner_loop_progress.set_postfix(
                    {
                        "lr": curr_lr,
                        "DLoss": np.around(curr_dloss / curr_count, 4),
                        "CLoss": np.around(curr_closs / curr_count, 4),
                        "TotalLoss": np.around(
                            (curr_dloss + curr_closs) / curr_count, 4
                        ),
                        "closs_weight": closs_weight,
                        "dloss_weight": dloss_weight,
                        "nll_loss": nll_loss,
                    }
                )
                inner_loop_progress.update(1)

            # Log training Loss
            log_dict = {}
            extra_dict = {}
            mloss = np.around(curr_dloss / curr_count, 4)
            gloss = np.around(curr_closs / curr_count, 4)
            total_loss = mloss + gloss
            if np.isnan(gloss):
                print("Fnding Nan in gaussian loss, skipping this update")
                print("If you see this a lot, the model has diverged")
                continue
            loss_dict = {
                "epoch": epoch + 1,
                "lr": curr_lr,
                "closs_weight": closs_weight,
                "dloss_weight": dloss_weight,
                "loss/c_loss": gloss,
                "loss/d_loss": mloss,
                "loss/total_loss": total_loss,
            }
            log_dict.update(loss_dict)

            # Log the learned noise schedules for numerical dimensions
            num_noise_dict = {}
            if (
                hasattr(self.diffusion.num_schedule, "rho")
                # Check if we have numerical points
                and self.diffusion.num_schedule.rho().numel() > 0
            ):
                if (
                    self.diffusion.num_schedule.rho().dim() > 0
                    and len(self.diffusion.num_schedule.rho()) > 1
                ):
                    num_noise_dict = {
                        f"num_noise/rho_col_{i}": value.item()
                        for i, value in enumerate(self.diffusion.num_schedule.rho())
                    }
                else:
                    num_noise_dict = {
                        "num_noise/rho": self.diffusion.num_schedule.rho().item()
                    }
            log_dict.update(num_noise_dict)
            cat_noise_dict = {}
            try:
                # Log the learned noise schedules for categlrical dimensions

                if (
                    self.diffusion.cat_schedule.k().dim() > 0
                    and len(self.diffusion.cat_schedule.k()) > 1
                ):
                    cat_noise_dict = {
                        f"cat_noise/k_col_{i}": value.item()
                        for i, value in enumerate(self.diffusion.cat_schedule.k())
                    }
                else:
                    cat_noise_dict = {
                        "cat_noise/k": self.diffusion.cat_schedule.k().item()
                    }
            except Exception as e:
                pass
            log_dict.update(cat_noise_dict)

            # Adjust learning rate
            if self.lr_scheduler == "reduce_lr_on_plateau":
                self.scheduler.step(total_loss)
            elif self.lr_scheduler == "anneal":
                self._anneal_lr(epoch)
            elif self.lr_scheduler == "fixed":
                pass
            else:
                raise NotImplementedError(
                    f"LR scheduler with name '{self.lr_scheduler}' is not implemented"
                )

            # Update EMA models
            ema_update(
                list(self.ema_model.parameters()),
                list(self.diffusion._denoise_fn.parameters()),
                None,
            )
            # TODO: Infer from 'is_learnable' or smth.
            if len(list(self.ema_num_schedule.parameters())) > 0:
                # only update when we have a learnable scheulde
                ema_update(
                    list(self.ema_num_schedule.parameters()),
                    list(self.diffusion.num_schedule.parameters()),
                    None,
                )
            if len(list(self.ema_cat_schedule.parameters())) > 0:
                ema_update(
                    list(self.ema_cat_schedule.parameters()),
                    list(self.diffusion.cat_schedule.parameters()),
                    None,
                )

            # Save ckpt base on the best training loss
            if total_loss < best_loss and self.curr_epoch > 4000:
                best_loss = total_loss
                to_remove = glob.glob(
                    os.path.join(self.model_save_path, f"best_model_*")
                )
                if to_remove:
                    os.remove(to_remove[0])
                state_dicts = {
                    "denoise_fn": self.diffusion._denoise_fn.state_dict(),
                    "num_schedule": self.diffusion.num_schedule.state_dict(),
                    "cat_schedule": self.diffusion.cat_schedule.state_dict(),
                }
                torch.save(
                    state_dicts,
                    Path(self.model_save_path)
                    / f"best_model_{np.round(total_loss,4)}_{epoch+1}.pt",
                )
                patience = 0
            else:
                patience += 1  # increment patience if best loss is not surpassed
            # Compute default model loss dict.
            model_loss_dict = self.compute_loss(prefix="")
            # Compute and log EMA model loss
            curr_model, curr_num_schedule, curr_cat_schedule = self.to_ema_model()
            ema_loss_dict = self.compute_loss(prefix="ema_")

            self.to_model(curr_model, curr_num_schedule, curr_cat_schedule)

            ema_total_loss = ema_loss_dict.get(
                "loss/sample/ema_total_loss", float("inf")
            )
            # Save the best ema ckpt
            if ema_total_loss < best_ema_loss and self.curr_epoch > 4000:
                best_ema_loss = ema_total_loss
                to_remove = glob.glob(
                    os.path.join(self.model_save_path, f"best_ema_model_*")
                )
                if to_remove:
                    os.remove(to_remove[0])
                state_dicts = {
                    "denoise_fn": self.ema_model.state_dict(),
                    "num_schedule": self.ema_num_schedule.state_dict(),
                    "cat_schedule": self.ema_cat_schedule.state_dict(),
                }
                torch.save(
                    state_dicts,
                    os.path.join(
                        self.model_save_path,
                        f"best_ema_model_{np.round(ema_total_loss,4)}_{epoch+1}.pt",
                    ),
                )

            # Evaluate Sample Quality
            if self.must_evaluate(epoch + 1):
                print_with_bar(
                    f"Routine Generation Evaluation every {self.check_val_every}, currently at epoch #{epoch + 1} total_loss={total_loss}."
                )
                self.evaluate(
                    epoch=epoch,
                    log_dict=log_dict,
                    extra=extra_dict,
                    checkpoint=True,
                    run_ema=True,
                )
            log_dict.update(ema_loss_dict)
            log_dict.update(model_loss_dict)
            # Submit logs
            self.logger.log(log_dict)

        end_time = time.time()
        print_with_bar(
            f"Ending Training Loop, total training time = {end_time - start_time}"
        )
        self.logger.log({"training_time": end_time - start_time})

    def evaluate(
        self, epoch, log_dict, extra, run_ema: bool = False, checkpoint: bool = False
    ):
        """
        Adapted evaluate function, using parallel .
        Args:
            epoch ():
            log_dict ():

        Returns:

        """
        # TODO: Move this back to TabDiffMetric, and make it an 'evaluation' server
        #   / spinner like object with a processing pool.

        start_time = time.time()
        if checkpoint:
            self.save_checkpoint(epoch)

        synth_queue = Queue()
        ITER_DICT = {"default": False}
        if run_ema:
            ITER_DICT["ema"] = True
        work_queue = Queue()
        save_queue = Queue()
        for generate_key, ema_flag in ITER_DICT.items():
            # work_queue.put((generate_key, ema_flag))

            synth_df = self.sample_synthetic(
                num_samples=self.metrics.real_data_size,
                ema=ema_flag,
            )
            synth_queue.put((generate_key, synth_df))
            save_queue.put((generate_key, synth_df))

        while synth_queue.qsize() > 0:
            (evaluate_key, synth_df) = synth_queue.get()
            ema_flag = ITER_DICT[evaluate_key]
            out_metrics, extra_metrics, _ = self.evaluate_generation(
                syn_df=synth_df,
                save_metric_details=True,
                plot_density=False,
                ema=ema_flag,
            )
            log_dict.update(out_metrics)
            extra.update(extra_metrics)
        new_time = time.time()
        print(f"[EVAL] Time elapsed     : {new_time - start_time}")

        while save_queue.qsize() > 0:
            (generate_key, write_df) = save_queue.get()
            original_columns = list(write_df.columns)
            # TODO: Add required meta-data columns
            write_df["model"] = generate_key
            # TODO: Add epoch / round meta-data column
            write_df["epoch"] = epoch + 1

            write_df = write_df[["epoch", "model"] + original_columns]
            # epoch, model_type
            write_path = (
                Path(self.result_save_path)
                / "parquet"
                / generate_key
                / f"generated.parquet"
            )
            if not write_path.parent.is_dir():
                write_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = write_path.is_file()
            write(
                write_path,
                write_df,
                mkdirs=None,
                compression="GZIP",
                append=file_exists,
            )

    def save_checkpoint(self, epoch_index, save_optimizer=True):
        step = epoch_index + 1
        save_path = Path(self.model_save_path) / f"model_{step}.pt"
        ema_model_save_path = Path(self.model_save_path) / f"ema_model_{step}.pt"

        state_dicts = {
            "denoise_fn": self.diffusion._denoise_fn.state_dict(),
            "num_schedule": self.diffusion.num_schedule.state_dict(),
            "cat_schedule": self.diffusion.cat_schedule.state_dict(),
        }
        if save_optimizer:
            state_dicts["optimizer_state"] = self.optimizer.state_dict()
        # Save main modle
        torch.save(
            state_dicts,
            save_path,
        )
        # Ema save-patjh
        torch.save(
            self.ema_model.state_dict(),
            ema_model_save_path,
        )

    def report_test(self, num_runs):
        base_dir = Path(self.result_save_path)

        shape_ = []
        trend_ = []
        mle_ = []
        c2st_ = []
        for i in range(num_runs):
            print_with_bar(f"GENERAL Evaluation Run {i}")
            # self.diffusion.num_timesteps = 50
            syn_df = self.sample_synthetic(
                num_samples=self.metrics.real_data_size, keep_nan_samples=False
            )
            out_metrics, extras, syn_df = self.evaluate_generation(
                syn_df=syn_df,
                ema=False,
            )
            print(f"Results of Run {i} are: \n{out_metrics}")
            shape_.append(out_metrics["density/Shape"])
            trend_.append(out_metrics["density/Trend"])
            mle_.append(out_metrics["mle"])
            c2st_.append(out_metrics["c2st"])
            # Save samples for quality evaluation
            metric_save_dir = base_dir / "all_samples"
            ensure_exists(metric_save_dir)
            syn_df.to_csv(metric_save_dir / f"samples_{i}.csv", index=False)

        shape_ = np.array(shape_)
        trend_ = np.array(trend_)
        mle_ = np.array(mle_)
        c2st_ = np.array(c2st_)

        shape_error = (1 - shape_) * 100
        trend_error = (1 - trend_) * 100
        c2st_percent = c2st_ * 100

        all_results = pd.DataFrame(
            {
                "shape": shape_error,
                "trend": trend_error,
                "mle": mle_,
                "c2st": c2st_percent,
            }
        )
        avg = all_results.mean(axis=0).round(3)
        std = all_results.std(axis=0).round(3)
        avg_std = pd.concat([avg, std], axis=1, ignore_index=True)
        avg_std.columns = ["avg", "std"]
        avg_std.index = [
            "shape",
            "trend",
            "mle",
            "c2st",
        ]

        # Savings
        all_results.to_csv(base_dir / "all_results.csv", index=True)
        avg_std.to_csv(base_dir / "avg_std.csv", index=True)
        print_with_bar(f"The AVG over {num_runs} runs are: \n{avg_std}")

    def report_test_dcr(self, num_runs):
        base_dir = Path(self.result_save_path)

        dcr_ = []
        dcr_real_ = []
        dcr_test_ = []

        for i in range(num_runs):
            print_with_bar(f"DCR Evaluation Run {i}")
            out_metrics, extras, syn_df = self.evaluate_generation()
            print(f"Results of Run {i} are: \n{out_metrics}")
            dcr_.append(out_metrics["dcr"])
            dcr_real_.append(extras["dcr_real"])
            dcr_test_.append(extras["dcr_test"])
            metric_save_dir = base_dir / "all_samples"
            ensure_exists(metric_save_dir)
            syn_df.to_csv(metric_save_dir / f"samples_{i}.csv", index=False)

        dcr_ = np.array(dcr_)

        dcr_percent = dcr_ * 100

        all_results = pd.DataFrame(
            {
                "dcr": dcr_percent,
            }
        )
        avg = all_results.mean(axis=0).round(3)
        std = all_results.std(axis=0).round(3)
        avg_std = pd.concat([avg, std], axis=1, ignore_index=True)
        avg_std.columns = ["avg", "std"]
        avg_std.index = [
            "dcr",
        ]

        # Savings
        all_results.to_csv(base_dir / "all_results.csv", index=True)
        avg_std.to_csv(base_dir / "avg_std.csv", index=True)
        dcr_real = np.concatenate(dcr_real_, axis=0)
        dcr_test = np.concatenate(dcr_test_, axis=0)
        dcr_df = pd.DataFrame({"dcr_real": dcr_real, "dcr_test": dcr_test})
        dcr_df.to_csv(base_dir / "dcr.csv", index=False)

        print_with_bar(f"The AVG over {num_runs} runs are: \n{avg_std}")

    def test(self):
        out_metrics, _, _ = self.evaluate_generation(
            save_metric_details=True, plot_density=False
        )
        print_with_bar(f"Results of the test are: \n{out_metrics}")
        self.logger.log(out_metrics)
        print(out_metrics)

    @torch.no_grad()
    def sample_synthetic(self, num_samples, keep_nan_samples=True, ema=False):
        if ema:
            curr_model, curr_num_schedule, curr_cat_schedule = self.to_ema_model()
        info = self.metrics.info

        print_with_bar(f"Starting Sampling, total samples to generate = {num_samples}")
        start_time = time.time()

        syn_data = self.diffusion.sample_all(
            num_samples,
            self.sample_batch_size,
            keep_nan_samples=keep_nan_samples,
            # batch=next(iter(self.train_iter)).to("cuda"),
        )
        print(f"Shape of the generated sample = {syn_data.shape}")

        if keep_nan_samples:
            num_all_zero_row = (syn_data.sum(dim=1) == 0).sum()
            if num_all_zero_row:
                print(
                    f"The generated samples contain {num_all_zero_row} Nan instances!!!"
                )
                self.logger.log({"num_Nan_sample": num_all_zero_row})

        # Recover tables'
        num_inverse = self.dataset.num_transform.inverse_transform
        int_inverse = self.dataset.int_transform.inverse_transform
        cat_inverse = self.dataset.cat_transform.inverse_transform
        syn_df = inverse_data_transform(
            info,
            syn_data,
            cat_inverse=cat_inverse,
            int_inverse=int_inverse,
            num_inverse=num_inverse,
        )

        end_time = time.time()
        print_with_bar(
            f"Ending Sampling, totoal sampling time = {end_time - start_time}"
        )

        if ema:
            self.to_model(curr_model, curr_num_schedule, curr_cat_schedule)

        return syn_df

    def to_ema_model(self):
        curr_model = self.diffusion._denoise_fn
        curr_num_schedule = self.diffusion.num_schedule
        curr_cat_schedule = self.diffusion.cat_schedule
        self.diffusion._denoise_fn = (
            self.ema_model
        )  # temporarily install the ema parameters into the model
        self.diffusion.num_schedule = self.ema_num_schedule
        self.diffusion.cat_schedule = self.ema_cat_schedule

        return curr_model, curr_num_schedule, curr_cat_schedule

    def to_model(self, curr_model, curr_num_schedule, curr_cat_schedule):
        self.diffusion._denoise_fn = curr_model  # give back the parameters
        self.diffusion.num_schedule = curr_num_schedule
        self.diffusion.cat_schedule = curr_cat_schedule

    def test_impute(
        self,
        X_0: torch.FloatTensor,
        trail_start,
        trial_size,
        resample_rounds,
        impute_condition,
        imputed_sample_save_dir: Path,
        w_num,
        w_cat,
        num_mask_indices: Optional[List[int]] = None,
        num_cat_indices: Optional[List[int]] = None,
    ):
        """

        Args:
            X_0 (): Original data tensor to run imputation on. N.B., this must not be masked yet.
            trail_start (): Starting time-step to run from
            trial_size (): Number of total time-steps to run with
            resample_rounds (): Number of annealing steps per tiem-step
            impute_condition (): COnditioning on noise or original sample.
            imputed_sample_save_dir ():
            w_num (): Omega (when y_only model is provided)
            w_cat (): Omega (when y_only model is provided)
            num_mask_indices (): List of indices to mask numerically
            num_cat_indices (): List of indices to mask numerically

        Returns:

        """

        # Note, this assumes imputation is done on the last epoch. This si wrong
        self.diffusion.eval()

        info = self.metrics.info
        task_type = info["task_type"]
        d_numerical, categories = self.dataset.d_numerical, self.dataset.categories
        if num_mask_indices is None and num_cat_indices is None:
            num_mask_idx, cat_mask_idx = [], []
        else:
            # Otherwise unpack the provided indices.
            num_mask_idx = num_mask_indices or []
            cat_mask_idx = cat_mask_idx or []
        # Construct
        x_num_train, x_cat_train = (
            self.dataset.X[:, :d_numerical],
            self.dataset.X[:, d_numerical:],
        )

        # TODO: Change to column type
        if task_type in {"binclass", "multiclass"}:
            # for cat cols, push the masked col to [MASK]
            cat_mask_idx += [0]
        else:
            # for num cols, set the masked col to the col mean (Quantile encoded with normal Distribution).
            num_mask_idx += [0]
            avg = x_num_train[:, num_mask_idx].mean(0).to(self.device)

        with torch.no_grad():

            for trial in range(trail_start, trail_start + trial_size):
                print_with_bar(f"Imputing trial {trial}")
                X_test = self.test_dataset.X
                X_test = deepcopy(X_test).to("cuda")

                X_num_impute, X_cat_impute = (
                    X_test[:, :d_numerical],
                    X_test[:, d_numerical:].long(),
                )

                # Apply mask to x_0
                if num_mask_idx:
                    X_num_impute[:, num_mask_idx] = avg
                if cat_mask_idx:
                    X_cat_impute[:, cat_mask_idx] = torch.tensor(
                        categories, dtype=X_cat_impute.dtype, device=X_cat_impute.device
                    )[cat_mask_idx]

                # Sample imputed tables
                impute_data = self.diffusion.sample_impute(
                    X_num_impute,
                    X_cat_impute,
                    num_mask_idx,
                    cat_mask_idx,
                    resample_rounds,
                    impute_condition,
                    w_num,
                    w_cat,
                )
                print(f"Shape of the imputed sample = {impute_data.shape}")

                # TODO: use function for this ,this is replicated too many times.
                # Recover tables
                num_inverse = self.dataset.num_transform
                int_inverse = self.dataset.int_transform
                cat_inverse = self.dataset.cat_transform

                if torch.any(
                    (impute_data[:, d_numerical + 1 :]).max(dim=0).values
                    > (x_cat_train[:, 1:]).max(dim=0).values
                ):
                    # if the test set contains categories not presented in the train set, we can not do cat_inverse.
                    # So we implement a patch that set those columns to the same as the train set
                    print(
                        "Test set contains extra categories, and so does imputed syn data. We cannot do cat_inverse. "
                        "So we set the cat columns as the same as the train set"
                    )
                    impute_data[:, d_numerical + 1 :] = x_cat_train[
                        : impute_data.shape[0], 1:
                    ]

                syn_num, syn_cat, syn_target = split_num_cat_target(
                    impute_data, info, num_inverse, int_inverse, cat_inverse
                )
                syn_df = recover_data(syn_num, syn_cat, syn_target, info)

                idx_name_mapping = info["idx_name_mapping"]
                idx_name_mapping = {
                    int(key): value for key, value in idx_name_mapping.items()
                }

                syn_df.rename(columns=idx_name_mapping, inplace=True)

                # Save imputed samples
                ensure_exists(imputed_sample_save_dir)
                syn_df.to_csv(f"{imputed_sample_save_dir}/{trial}.csv", index=False)
                print(
                    f"Imputed samples are saved to {imputed_sample_save_dir}/{trial}.csv"
                )

    def must_evaluate(self, epoch) -> bool:
        """
        Returns whether we must evaluate during this round...
        Args:
            epoch ():

        Returns:

        """
        return (epoch + 1) % self.check_val_every == 0
