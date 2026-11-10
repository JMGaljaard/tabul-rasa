from typing import Callable, Optional, Tuple

import omegaconf
import torch
from overrides import overrides
from torch import nn
from torch.nn import functional as F

from tabdiff.models import UnifiedCtimeDiffusion, UnifiedDtimeDiffusion
from tabdiff.models.modules import TabDiffDenoiseFn, UniModMLP
from tabdiff.models.trainer import init_from_checkpoint, unpack_mixed_loss
from tabdiff.models.unified_ctime_diffusion import (
    MixedDiffusionModelOutput,
    UnifiedDiffusionOutput,
)
from tabdiff.unlearn.callback import NegativePreferenceOptimization, UnlearnCallback
from .untrainer import UnTrainer
from tabdiff.models.schedulers import ScaleSchedule


class AlignTrainer(UnTrainer):
    """
    Alignment trainer for aligning (or un-aligning) a model with itself / differently.
    In addition, it constructs a reference model that is used to do preference based optimizaiotn
    for unlearning.
    """

    #
    # @overrides
    # def must_evaluate(self, epoch) -> bool:
    #     """
    #     Returns whether we must evaluate during this round, early rounds are always evaluated.
    #     Args:
    #         epoch ():
    #
    #     Returns:
    #
    #     """
    #     return (epoch + 1) % self.check_val_every == 0

    @classmethod
    def from_config_(
        cls,
        config,
        *args,
        **kwargs,
    ):
        backbone = UniModMLP.from_config(config.diffusion.unimodmlp_params)
        denoise_fn = TabDiffDenoiseFn.from_config(
            config.diffusion.edm_params, denoise_fn=backbone
        )
        # fixme: from config
        denoise_fn.to("cuda")
        if "ddpm" in config.diffusion.exp_name_model:
            reference_diffusion = UnifiedDtimeDiffusion.from_config(
                config.diffusion,
                denoise_fn=denoise_fn,
                y_only_model=None,
            )
        else:
            reference_diffusion = UnifiedCtimeDiffusion.from_config(
                config.diffusion,
                denoise_fn=denoise_fn,
                y_only_model=None,
            )

        reference_diffusion.to("cuda")
        reference_diffusion.eval()
        init_from_checkpoint(
            reference_diffusion, kwargs.get("ckpt_path"), kwargs.get("device")
        )
        closs_scaler = ScaleSchedule.from_config(
            config.unlearn.main.loss_schedules.continuous,
        )
        dloss_scaler = ScaleSchedule.from_config(
            config.unlearn.main.loss_schedules.discrete,
        )
        return cls(
            config,
            *args,
            **kwargs,
            reference_model=reference_diffusion,
            closs_schedule=closs_scaler,
            dloss_schedule=dloss_scaler,
        )

    def __init__(
        self,
        config: omegaconf.DictConfig | omegaconf.OmegaConf,
        *args,
        reference_model: UnifiedCtimeDiffusion | UnifiedDtimeDiffusion,
        **kwargs,
    ):

        super().__init__(
            config,
            *args,
            **kwargs,
        )
        self._ref_model = reference_model
        self.curr_epoch = 0

    def _compute_reference_loss(
        self,
        mixed_output: MixedDiffusionModelOutput,
        x0: torch.FloatTensor,
        debug: bool = False,
    ) -> (MixedDiffusionModelOutput, MixedDiffusionModelOutput):
        """
        Compute reference models' ouptu follwoing th e
        Args:
            mixed_output (MixedDiffusionModelOutput): Output of the negative direction
            x0 ():
            debug (bool): Enable asserts during dev.
        Returns:

        """
        learn_num_loss, learn_cat_loss = None, None
        x_num_t, x_cat_t = None, (None, None)
        y_num_t, y_num_t = None, None
        if has_num := mixed_output.continuous_output is not None:
            x_num_t = mixed_output.continuous_output.xt
            y_num_t = mixed_output.continuous_output.target
            # num_weight = mixed_output.continuous_output.weight
        if has_cat := mixed_output.categorical_output is not None:
            x_cat_t = mixed_output.categorical_output.xt
            x_cat_t_soft = mixed_output.categorical_output.xt_soft
            y_cat_t = mixed_output.categorical_output.target
            # cat_weight = mixed_output.categorical_output.weight

        t = mixed_output.time
        # Note, `x` is only used a result for the
        dsigma_cat, sigma_cat, sigma_num, _ = self._ref_model.sample_noise(x_num_t, t=t)
        is_learnable = self.diffusion.cat_scheduler == "log_linear_per_column"
        strategy = "soft" if is_learnable else "hard"
        # fixme: Use actual implementation with overwritten values instead of manual...
        pref_model_out_num, pref_model_out_cat = self._ref_model._denoise_fn(
            x_num_t,
            x_cat_t_soft if strategy == "soft" else x_cat_t,
            t.squeeze(),
            sigma=sigma_num,
        )

        ref_cat_loss, ref_num_loss = self.compute_mixed_loss(
            mixed_output=mixed_output,
            model_cat_logit=self._ref_model._subs_parameterization(
                pref_model_out_cat, x_cat_t
            ),
            model_num_pred=pref_model_out_num,
            model=self._ref_model,
        )

        reference_output = MixedDiffusionModelOutput(
            categorical_output=ref_cat_loss,
            continuous_output=ref_num_loss,
            time=t,
        )

        if debug:

            learner_cat_loss, learner_num_loss = self.compute_mixed_loss(
                mixed_output=mixed_output,
                model_cat_logit=mixed_output.categorical_output.logit,
                model_num_pred=mixed_output.continuous_output.logit,
                model=self.diffusion,
            )
            learner_output = MixedDiffusionModelOutput(
                categorical_output=learner_cat_loss,
                continuous_output=learner_num_loss,
                time=t,
            )
            assert torch.all(
                torch.eq(
                    learner_output.categorical_output.loss
                    * mixed_output.categorical_output.weight,
                    mixed_output.categorical_output.loss,
                )
            ), "Categorical noise of learner is not correctly reconstructed"
            assert torch.all(
                torch.eq(
                    learner_output.continuous_output.loss
                    * mixed_output.continuous_output.weight,
                    mixed_output.continuous_output.loss,
                )
            ), "Numerical noise of learner is not correctly reconstructed"
        return reference_output

    def compute_mixed_loss(
        self,
        mixed_output: MixedDiffusionModelOutput,
        model_cat_logit: torch.FloatTensor,
        model_num_pred: torch.FloatTensor,
        model: UnifiedCtimeDiffusion,
    ):
        ref_num_loss = None
        ref_cat_loss = None

        x_num_t, x_cat_t = None, (None, None)
        y_num_t, y_num_t = None, None
        if has_num := mixed_output.continuous_output is not None:
            x_num_t = mixed_output.continuous_output.xt
            y_num_t = mixed_output.continuous_output.target
            # num_weight = mixed_output.continuous_output.weight
        if has_cat := mixed_output.categorical_output is not None:
            x_cat_t = mixed_output.categorical_output.xt
            x_cat_t_soft = mixed_output.categorical_output.xt_soft
            y_cat_t = mixed_output.categorical_output.target
            # cat_weight = mixed_output.categorical_output.weight

        dsigma_cat, sigma_cat, sigma_num, _ = model.sample_noise(
            x_num_t, t=mixed_output.time
        )

        if has_num:
            # Hijack callback usage to compute unweighted loss
            # MSE loss
            ref_num_loss = model._edm_loss(
                D_yn=model_num_pred,
                xt=x_num_t,
                y=y_num_t,
                sigma=sigma_num,
                callback=lambda D_yn, target, xt, weight: (D_yn - target) ** 2,
            )
        if has_cat:
            # Make sure to set the logits to .... (zero)
            # Logsoftmax loss
            # TODO: This is not yet masked?
            ref_cat_loss = model._absorbed_closs(
                model_output=model_cat_logit,
                x0=y_cat_t,
                xt=x_cat_t,
                sigma=sigma_cat,
                dsigma=dsigma_cat,
                callback=lambda D_yn, target, xt, weight: torch.gather(
                    D_yn, -1, target[:, :, None]
                ).squeeze(-1),
            )
        return ref_cat_loss, ref_num_loss

    @overrides
    def _run_step(self, x, closs_weight, dloss_weight, epoch: int = -1):
        """

        Args:
            x (): Sample at X0
            closs_weight (): Continuous loss scaling
            dloss_weight (): Categorical loss scaling.

        Returns:

        """
        if isinstance(x, torch.Tensor):
            sample = x.to(self.device, non_blocking=True)
        else:
            sample = x["perturbed"].to(self.device, non_blocking=True)
            original_sample = x["sample"].to(self.device, non_blocking=True)
        self.diffusion.train()

        self.optimizer.zero_grad()

        # Compute the original loss (MSE and log-softmax'd (crossentropy))
        learner_output: MixedDiffusionModelOutput = self.diffusion.mixed_loss(
            sample,
            num_callback=None,
            cat_callback=None,
        )

        learner_cat_loss, learner_num_loss = (
            learner_output.categorical_output.loss,
            learner_output.continuous_output.loss,
        )

        # Compute reference output
        reference_output = None
        with torch.no_grad():
            if self._ref_model is not None:
                reference_output = self._compute_reference_loss(
                    learner_output, x0=sample
                )

        align_loss = self._compute_mixed_align_loss(
            mixed_output=learner_output,
            reference_output=reference_output,
            return_ratio=True,
        )
        # # V07 add loss scaling
        # loss_scale = (
        #     (self._config.unlearn.main.steps - epoch) / self._config.unlearn.main.steps
        # ) * 0.95 + 0.05
        #
        # loss_scale = min(1, max(loss_scale, 0.05))
        # reg_scale = 1 - loss_scale
        # For V06, instead we will use the ratio's, and comptue a joint distributoin
        # for the log-softmax. Lke V02, we do not et
        # Here, we unpack the losses, and compute the reaito

        num_beta, cat_beta = (
            align_loss.continuous_output.weight,
            align_loss.categorical_output.weight,
        )

        c_ratio, d_ratio_masked = unpack_mixed_loss(
            align_loss,
            # TODO:
            #   Expliciltyt set to -1 for now, s.t. everything is unmasked
            #   Maybe we shoudl account for 'masked' ness
            # No, this breaks
            mask_index=None,
            reduce=False,
            # Same as NPO2
            scale=False,
            device=self.device,
        )

        # Reweight (optionally) the losses for optimization
        # Note: needs ot be mad e ascalar
        # NOTE: we perform disjoint optimization, as joint leads to nothing
        loss = (
            (num_loss := -F.logsigmoid(-num_beta * (c_ratio)).sum(dim=1).mean())
            + (
                cat_loss := -F.logsigmoid(-cat_beta * (d_ratio_masked))
                .sum(dim=1)
                .mean()
            )
        ) * 2  # * loss_scale
        # if (epoch + 1) % 20 == 0:
        #     print(loss_scale)
        # else:
        #     closs, dloss = unpack_mixed_loss(
        #         align_loss,
        #         # TODO:
        #         #   Expliciltyt set to -1 for now, s.t. everything is unmasked
        #         #   Maybe we shoudl account for 'masked' ness
        #         # No, this breaks
        #         mask_index=None,
        #         reduce=True,
        #         # Same as NPO2
        #         scale=False,
        #         device=self.device,
        #     )
        #
        #     loss = dloss_weight * dloss + closs_weight * closs

        # Since V04/05, add explicit regularization with the orindary retain loss using only the
        # masked samples to 'prefer' the unbiased distribution.
        # is_masked = (learner_output.categorical_output.xt == self.diffusion.mask_index)[
        #     :, self._update_fn_cat.feature_index
        # ]
        # Compute normal (time) weighted loss.
        # normal_closs, normal_dloss = (
        #     (learner_num_loss * learner_output.continuous_output.weight)[
        #         is_masked.flatten()
        #     ].mean(dim=1),
        #     (learner_cat_loss * learner_output.categorical_output.weight)[
        #         is_masked.flatten()
        #     ].mean(dim=1),
        # )

        # V04 Also add regularization with normal loss on unconditional samples.
        # loss += (
        #     # self._config.unlearn.reg_lambda
        #     # Use time-weighted regularization loss
        #     reg_scale
        #     * (closs_weight * normal_closs + dloss_weight * normal_dloss).mean()
        # )
        loss.backward()
        # Clip gradients by default as it is expected to prevent divergence
        # nn.utils.clip_grad_norm_(
        #     self.diffusion.parameters(),
        #     1.0,
        #     norm_type=2.0,
        # )
        self.optimizer.step()
        return (
            num_loss.mean().item(),
            cat_loss.mean().item(),
        )

    def _compute_dtype_align_loss(
        self,
        learn_output: MixedDiffusionModelOutput,
        reference_output: MixedDiffusionModelOutput,
        update_fn: Optional[NegativePreferenceOptimization] = None,
        align_loss_fn: Callable[[torch.Tensor, ...], torch.Tensor] = None,
        is_numeric: bool = None,
        xt_joint: Tuple[torch.Tensor] = None,
        is_positive_sample: torch.Tensor = None,
        return_ratio: bool = False,
    ) -> UnifiedDiffusionOutput:
        """
        Compute loss of aligning models, uses callback function to align positively / negatively
        Args:
            learn_output (): Predicted (EDM formulation) of the learner / changed model.
            reference_output (): Reference (EDM formulation) of the reference / align model.
            update_fn (): Alignment function
            align_loss_fn (): Loss function for alignment.

        Returns:

        """
        align_loss = None
        align_ratio = None
        reference_loss = None
        xt = None
        weight = None
        if learn_output is not None and reference_output is not None:
            learner_loss = learn_output.loss
            reference_loss = reference_output.loss
            weight = learn_output.weight

            xt = reference_output.xt
            if update_fn is not None:
                xt = reference_output.xt
                # Automatically perform the update with the Callback.
                #   Note, assumes that the method is provided with a Gradient Ascent module...
                # Compute the ratio of logprob deviation of the two models.

                # Hijack code to perform the alignment ratio, using subtraction
                align_ratio = update_fn.forward(
                    D_yn=learner_loss,
                    x0=reference_loss,
                    xt=xt,
                    weight=None,
                    loss_fn=align_loss_fn,
                    mask_index=(None if is_numeric else self.diffusion.mask_index),
                    is_positive_sample=is_positive_sample,
                )
                # Scale the ratio by SNR.
            else:
                align_ratio = align_loss_fn(learner_loss, reference_loss)
            #
            # # Multiply within log-sigmoid...
            # align_ratio *= weight
        dtype_beta = update_fn.get_beta(align_ratio, is_numeric=is_numeric)

        if align_ratio is not None and return_ratio:
            # Otherwise, we return the (log) ratio itself.
            return UnifiedDiffusionOutput(
                loss=align_ratio,
                target=None,
                logit=None,
                weight=dtype_beta,
                xt=xt,
            )

        elif align_ratio is not None:
            # Retrieve beta  from the loss_fn itself

            #  TODO: Does the NPO loss know about the index? (DONOTCOMMIT)
            inner_beta_coeff = dtype_beta
            outer_beta_coeff = 2.0  # / dtype_beta
            # TODO: Undo for NPO
            align_loss = (
                -F.logsigmoid(
                    -inner_beta_coeff
                    * (
                        align_ratio
                        # .mean(dim=1)
                    )
                )
                * outer_beta_coeff
            )

            return UnifiedDiffusionOutput(
                loss=align_loss,
                target=reference_loss,
                logit=None,
                weight=weight,
                xt=xt,
            )

    def _compute_mixed_align_loss(
        self,
        mixed_output: MixedDiffusionModelOutput,
        reference_output: Optional[MixedDiffusionModelOutput],
        return_ratio: bool = False,
    ):
        """
        Compute mixed alignment loss for different dtypes.
        Args:
            mixed_output ():
            reference_output ():

        Returns:

        """
        cat_learn, cat_ref = (
            mixed_output.categorical_output,
            reference_output.categorical_output,
        )
        xt_num, xt_cat = (
            mixed_output.categorical_output.xt,
            mixed_output.categorical_output.xt,
        )
        num_learn, num_ref = None, None
        if reference_output is not None:
            num_learn, num_ref = (
                mixed_output.continuous_output,
                reference_output.continuous_output,
            )

        align_loss_fn = torch.sub

        # Determine positive or negative 'ratio'
        if self._config.unlearn.dtype == "numerical":
            is_positive_sample = 1
        else:
            is_unmasked = (xt_cat != self.diffusion.mask_index)[
                :, [self._update_fn_cat.column]
            ]
            is_positive_sample = (-1) ** (is_unmasked.float())
        cat_align_loss = self._compute_dtype_align_loss(
            cat_learn,
            cat_ref,
            update_fn=self._update_fn_cat,
            align_loss_fn=align_loss_fn,
            is_numeric=False,
            is_positive_sample=is_positive_sample,
            return_ratio=return_ratio,
        )
        # This is effectively neg-grad applied on the 'unlearn' column...
        #   Test: make sure that we are correctly optimizng???
        num_align_loss = self._compute_dtype_align_loss(
            num_learn,
            num_ref,
            update_fn=self._update_fn_num,
            align_loss_fn=align_loss_fn,
            is_numeric=True,
            is_positive_sample=is_positive_sample,
            return_ratio=return_ratio,
        )

        return MixedDiffusionModelOutput(
            categorical_output=cat_align_loss,
            continuous_output=num_align_loss,
        )
