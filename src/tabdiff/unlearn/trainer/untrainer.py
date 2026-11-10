import torch

from torch import nn

from tabdiff.models import Trainer
from tabdiff.models.schedulers import AnnealSchedule, ConstantSchedule, ScaleSchedule
from tabdiff.models.trainer import unpack_mixed_loss
from tabdiff.models.unified_ctime_diffusion import MixedDiffusionModelOutput
from tabdiff.unlearn.callback import UnlearnCallback

from tabdiff.utils import get_encoded_indices


class UnTrainer(Trainer):
    """
    Unlearning trainer as defined by ...
    """

    def must_evaluate(self, epoch) -> bool:
        """
        Returns whether we must evaluate during this round, early rounds are always evaluated.
        Args:
            epoch ():

        Returns:

        """
        return (epoch < 10) or ((epoch + 1) % self.check_val_every == 0)

    @classmethod
    def from_config_(cls, config, *args, **kwargs):
        # For default reasons, we also provide backwards compatibility by defaulting ot paper values
        steps = (
            config.unlearn.main.steps
            if (
                hasattr(config, "unlearn")
                and hasattr(config.train, "main")
                and hasattr(config.train, "steps")
            )
            else 500
        )

        closs_scale_or_default = AnnealSchedule(loss_lambda=1.0, total_steps=steps)
        dloss_scale_or_default = ConstantSchedule(loss_lambda=1.0)

        try:
            closs_scale_or_default = ScaleSchedule.from_config(
                config.unlearn.main.loss_schedules.continuous,
            )
        except Exception as e:
            print(f"Defaulting to paper defaults continuout: steps={steps} scale=1.0")
        try:
            dloss_scale_or_default = ScaleSchedule.from_config(
                config.unlearn.main.loss_schedules.discrete,
            )
        except Exception as e:
            print(f"Defaulting to paper defaults continuout: steps={steps} scale=1.0")

        evaluation_parallelism = (
            10
            if hasattr(config.experiment, "evaluation_parallelism")
            else config.experiment.evaluation_parallelism
        )
        return UnTrainer(
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
        *args,
        update_fn_num: UnlearnCallback,
        update_fn_cat: UnlearnCallback,
        **kwargs,
    ):

        super().__init__(config, *args, **kwargs)
        self._update_fn_num = update_fn_num
        self._update_fn_cat = update_fn_cat
        # Overwrite behavior of super-constructor.
        self.curr_epoch = 0

    def evaluate(
        self,
        epoch,
        log_dict,
        extra,
        **kwargs,
    ):
        """
        Adapted evaluate function. Skipping checkpointing.

        Notes: By default, we disable checkpointing, to save space during evaluation.
        Args:
            epoch (int): Epoch to be evaluated on (context)
            log_dict (tp.Dict): Config object ot keep track of the evaluation of (unlearning/utility) metrics during
                unlearning / knowledge editing.

        Returns:

        """
        # Perform 'normal' metrics
        # TODO: Make that we can continue with evaluation of the metrics, such that we can store them additionally.
        super().evaluate(epoch, log_dict, extra, run_ema=False, checkpoint=False)

        # Perform 'unlearn' metrics/

        # TODO:
        #   1. Impute values (unshuffled) for test / validation set
        #   2. Compute reconstruction accurayc (MAE / ... )
        #   3. Store metrics in appropriate location
        #
        #
        #
        #
        # ul_config_indices = self._config.unlearn.column
        # num_ul_indices, cat_ul_indices = get_encoded_indices(
        #     ul_config_indices,
        #     info=self.dataset.info,
        #     reverse=False,
        #     split=True,
        # )
        #
        # for split, dataset in {"train": self.dataset, "test": self.test_dataset}.items():
        #     # Get original data
        #     X_0: torch.FloatTensor = dataset.X.clone()
        #
        #     impute_condition = "x_0"
        #
        #     reconstructed_sample = self.test_impute(
        #         X_0,
        #         1,  # Ultimate step
        #         self._config.diffusion.num_timesteps,  # From config
        #         impute_contition="X_0",
        #         imputed_sample_save_dir=None,
        #         resample_rounds=1,
        #         w_num=0,
        #         w_cat=0,
        #         num_mask_indices=num_ul_indices,
        #         num_cat_indices=cat_ul_indices,
        #     )
        #     # TODO: Comptue reconstruction accuracy.

    def _run_step(self, x, closs_weight, dloss_weight, epoch: int = -1):
        """
        Perform forward / backward pass with model in training fashion,
        Args:
            x (torch.FloatTensor): Encoded data tensor with tabular rcords.
            closs_weight (): Continuous loss scaling weight according to loss-scaling (i.e., Linear / Annealing loss).
            dloss_weight (): Discrete loss scaling weight according to loss-scaling (i.e., Linear / Annealing loss).

        Returns:
            float: Loss of discrete component.
            float: Loss of continuous component.
        """
        if isinstance(x, torch.Tensor):
            x_sample = x.to(self.device, non_blocking=True)
        else:
            if "perturbed" not in x:
                x_sample = x["sample"].to(self.device, non_blocking=True)
            else:
                x_sample = x["perturbed"].to(self.device, non_blocking=True)

        self.diffusion.train()

        self.optimizer.zero_grad()

        mixed_output: MixedDiffusionModelOutput = self.diffusion.mixed_loss(
            x_sample,
            num_callback=self._update_fn_num,
            cat_callback=self._update_fn_cat,
        )

        closs, dloss = unpack_mixed_loss(
            mixed_output,
            self.diffusion.mask_index,
            reduce=True,
            scale=True,
            device=self.device,
        )
        loss = dloss_weight * dloss + closs_weight * closs
        loss.backward()

        # Clip gradients by default, to prevent divergence.
        nn.utils.clip_grad_norm_(
            self.diffusion.parameters(),
            1.0,
            norm_type=2.0,
        )
        self.optimizer.step()

        return dloss.item(), closs.item()
