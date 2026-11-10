from typing import Callable, Optional

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
from .align_trainer import AlignTrainer
from tabdiff.models.schedulers import ScaleSchedule
from ...utils import get_encoded_indices


class BiAlignTrainer(AlignTrainer):
    """
    Bi-Alignment trainer for aligning (or un-aligning) a model with itself / differently.
    In addition, it constructs a reference model that is used to do preference based optimizaiotn
    for unlearning.
    """

    def must_evaluate(self, epoch) -> bool:
        return (epoch + 1) % self.check_val_every == 0

    @classmethod
    def from_config_(
        cls,
        config,
        *args,
        **kwargs,
    ):
        reference_diffusion = None

        # Only load the refernece model if it is requested by the learner.
        if config.unlearn.reference_model:
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
            reference_model=reference_model,
        )

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
            ref_num_loss = model._edm_loss(
                D_yn=model_num_pred,
                xt=x_num_t,
                y=y_num_t,
                sigma=sigma_num,
                callback=lambda D_yn, target, xt, weight: (D_yn - target) ** 2,
            )
        if has_cat:
            # Make sure to set the logits to .... (zero)

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

    def _run_step(self, x, closs_weight, dloss_weight):
        """

        Args:
            x (): Sample at X0
            closs_weight (): Continuous loss scaling
            dloss_weight (): Categorical loss scaling.

        Returns:

        """
        if isinstance(x, torch.Tensor):
            x = x.to(self.device, non_blocking=True)
        else:
            _ = x["perturbed"].to(self.device, non_blocking=True)
            x = x["sample"].to(self.device, non_blocking=True)

        # STep 1: Perform data-multilaiont outself
        x_cat, x_num = self.diffusion.split_input(x)

        # Pre-compute indices of columns to perform alignment on.
        num_indices, cat_indices = get_encoded_indices(
            self._config.unlearn.column, self.dataset.info, reverse=False, split=True
        )

        # Step 2: Create 'win' scenario, this one will no
        dsigma_cat_shared, sigma_cat_shared, sigma_num_shared, shared_time = (
            self.diffusion.sample_noise(x, t=None)
        )

        # [MASKED] SAMPLE CREATION, i.e. "WIN" Sample
        # Perturb win sample, this one has the condition masked (TODO: figure out Numerical)_
        move_chance_shared, x_cat_t_win_soft, x_cat_t_win, x_num_t_win = (
            self.diffusion.perturb_sample(
                sigma_cat_shared, sigma_num_shared, x_cat, x_num
            )
        )
        # Copy for ...
        move_chance_win = move_chance_shared.clone()
        # Alter win
        if x_num.numel() > 0 and len(num_indices) > 0:
            raise NotImplementedError("Numerical is not yet implemented")

        if x_cat.numel() > 0 and len(cat_indices) > 0:
            # We artificially perform masking by setting the transition likelihood extremely high.
            copy_mask_token = self.diffusion.mask_index[cat_indices]
            # Always change the target index.
            move_chance_win[:, copy_mask_token] = 1.0
            is_learnable = False
            strategy = "soft" if is_learnable else "hard"
            x_cat_t_win, x_cat_win_soft = self.diffusion.q_xt(
                x_cat,
                move_chance_win,
                strategy=strategy,
            )

        # [UNMASKED] SAMPLE CREATION, i.e. "LOSE" Sample
        # Use same-input.
        x_cat_t_lose, x_num_t_lose = (
            x_cat_t_win.clone(),
            x_num_t_win.clone(),
        )

        if x_num.numel() > 0 and len(num_indices) > 0:
            raise NotImplementedError("Numerical is not yet implemented")
        # Soft x_cat is the problem, but why?

        if x_cat.numel() > 0 and len(cat_indices) > 0:
            # We artificially perform unmasking by setting the transition likelihood extremely low.
            copy_mask_token = self.diffusion.mask_index[cat_indices]
            move_chance_lose = torch.zeros_like(move_chance_shared) - 1
            # We forcibly move the thingy.
            x_cat_t_lose[:, copy_mask_token] = x_cat[:, copy_mask_token]
            # Lastly, obtian a soft sample
            is_learnable = self.diffusion.cat_scheduler == "log_linear_per_column"
            strategy = "soft" if is_learnable else "hard"
            # Only to compute soft max
            x_cat_t_lose, x_cat_t_lose_soft = self.diffusion.q_xt(
                x_cat_t_lose,
                move_chance_lose,
                strategy="hard",
            )
        # TODO: Assertiions?
        self.diffusion.train()

        self.optimizer.zero_grad()

        # Compute the original loss on masked target column
        learner_win_loss: MixedDiffusionModelOutput = self.diffusion.mixed_loss(
            x,
            x_t=(move_chance_win, x_cat_t_win_soft, x_cat_t_win, x_num_t_win),
            time=(dsigma_cat_shared, sigma_cat_shared, sigma_num_shared, shared_time),
            num_callback=None,
            cat_callback=None,
        )
        # Compute original loss on unmasked target column
        learner_lose_loss: MixedDiffusionModelOutput = self.diffusion.mixed_loss(
            x,
            x_t=(move_chance_win, x_cat_t_lose_soft, x_cat_t_lose, x_num_t_win),
            time=(dsigma_cat_shared, sigma_cat_shared, sigma_num_shared, shared_time),
            num_callback=None,
            cat_callback=None,
        )

        # Compute same losses for the reference model (if available)

        reference_win_output, reference_lose_output = None, None
        with torch.no_grad():
            if self._ref_model:
                reference_win_output = self._ref_model.mixed_loss(
                    x,
                    x_t=(move_chance_win, x_cat_t_win_soft, x_cat_t_win, x_num_t_win),
                    time=(
                        dsigma_cat_shared,
                        sigma_cat_shared,
                        sigma_num_shared,
                        shared_time,
                    ),
                    num_callback=None,
                    cat_callback=None,
                )
            reference_lose_output = self._ref_model.mixed_loss(
                x,
                x_t=(
                    move_chance_win,
                    x_cat_t_lose_soft,
                    x_cat_t_lose,
                    x_num_t_win,
                ),
                time=(
                    dsigma_cat_shared,
                    sigma_cat_shared,
                    sigma_num_shared,
                    shared_time,
                ),
                num_callback=None,
                cat_callback=None,
            )

        num_loss, cat_loss = self._compute_mixed_align_loss(
            win_output=learner_win_loss,
            lose_output=learner_lose_loss,
            reference_win_output=reference_win_output,
            refernece_lose_output=reference_lose_output,
            cross_wise=True,
        )

        closs, dloss = num_loss.mean(), cat_loss.mean()

        # Reweight (optionally) the losses for optimization (scaling by epoch, not time step).
        loss = dloss_weight * dloss + closs_weight * closs
        # Clip gradients by default as it is expected to di
        nn.utils.clip_grad_norm_(
            self.diffusion.parameters(),
            1.0,
            norm_type=2.0,
        )
        loss.backward()
        self.optimizer.step()

        return dloss.item(), closs.item()

    def _compute_dtype_align_loss(
        self,
        learn_output: MixedDiffusionModelOutput,
        reference_output: MixedDiffusionModelOutput,
        update_fn: Optional[NegativePreferenceOptimization] = None,
        align_loss_fn: Callable[[torch.Tensor, ...], torch.Tensor] = torch.sub,
        is_numeric: bool = None,
        raw: bool = False,
    ) -> UnifiedDiffusionOutput:
        """
        Compute loss of aligning models, uses callback function to align positively / negatively
        Args:
            learn_output (): Predicted (EDM formulation) of the learner / changed model.
            reference_output (): Reference (EDM formulation) of the reference / align model.
            update_fn (): Alignment function
            align_loss_fn (): Loss function for alignment.
            raw (bool, False): Returns raw ratio, instead of post-activiatino loss.
        Returns:

        """
        align_loss = None
        align_ratio = None
        reference_loss = None
        xt = None
        weight = None
        # In case there is a reference model.
        if reference_output is not None:
            reference_loss = reference_output.loss

        # Compute (ratio) with learner
        if learn_output is not None:
            learner_loss = learn_output.loss

            weight = learn_output.weight

            xt = learn_output.xt
            if update_fn is not None:
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
                )
                # Scale the ratio by SNR.
            else:
                align_ratio = align_loss_fn(learner_loss, reference_loss)
            #
            # # Multiply within log-sigmoid...
            # align_ratio *= weight
        if raw:
            return align_ratio

        if align_ratio is not None:
            # Retrieve beta  from the loss_fn itself

            #  TODO: Does the NPO loss know about the index? (DONOTCOMMIT)
            beta_coeff = update_fn.get_beta(align_ratio, is_numeric=is_numeric)
            # Compute the objective through
            align_loss = -F.logsigmoid(beta_coeff * align_ratio).mul(
                2.0 / update_fn.get_beta(align_ratio, is_numeric=is_numeric)
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
        win_output: MixedDiffusionModelOutput,
        lose_output: MixedDiffusionModelOutput,
        reference_win_output: MixedDiffusionModelOutput,
        refernece_lose_output: MixedDiffusionModelOutput,
        cross_wise: bool = False,
    ):
        """
        Compute mixed alignment loss for different dtypes.
        Args:
            mixed_output ():
            reference_output ():
            cross_wise (bool): Perofrm 'normal' type of loss, else do cross-wise (09-02-2026)
        Returns:

        """
        # Initialize accumulators for fraciton
        num_accumulate = 0
        cat_accumulate = 0
        # Data is log-transformed (log softmax) / EDM formulate models, so use subtraciotn.
        align_loss_fn = torch.sub
        # Compute the locations at which input positions of the models were masked.
        win_is_masked = (
            win_output.categorical_output.xt == self.diffusion.mask_index
        ).float()
        lose_is_masked = (
            lose_output.categorical_output.xt == self.diffusion.mask_index
        ).float()

        # Get weights to use wihtin softmax, as not every time-step is as important.
        omega_t_num = win_output.continuous_output.weight
        omega_t_cat = win_output.categorical_output.weight
        beta_num, beta_cat = 1, 1
        num_loss = 0
        cat_loss = 0
        if not cross_wise:
            for mixed_output, reference_output, sign, mask in [
                (win_output, reference_win_output, 1, win_is_masked),
                (lose_output, refernece_lose_output, -1, lose_is_masked),
            ]:
                num_ref, cat_ref = None, None
                num_learn, cat_learn = (
                    mixed_output.continuous_output,
                    mixed_output.categorical_output,
                )
                if reference_output:
                    num_ref, cat_ref = (
                        reference_output.continuous_output,
                        reference_output.categorical_output,
                    )

                # THis call computes the NPO'd ratio, i.e. (cat_learn - cat_ref), with the forget column set set to -(cat_learn - cat_ref) = (cat_ref + cat_learn)
                cat_align_ratio = self._compute_dtype_align_loss(
                    cat_learn,
                    cat_ref,
                    update_fn=self._update_fn_cat,
                    align_loss_fn=align_loss_fn,
                    is_numeric=False,
                    raw=True,
                )
                # This is effectively neg-grad applied on the 'unlearn' column...
                #   Test: make sure that we are correctly optimizng???
                num_align_ratio = self._compute_dtype_align_loss(
                    num_learn,
                    num_ref,
                    update_fn=self._update_fn_num,
                    align_loss_fn=align_loss_fn,
                    is_numeric=True,
                    raw=True,
                )

                beta_num = self._update_fn_cat.get_beta(
                    num_align_ratio, is_numeric=True
                )
                beta_cat = self._update_fn_num.get_beta(
                    cat_align_ratio, is_numeric=False
                )

                # Update accumulators
                num_accumulate += sign * (beta_num * num_align_ratio)

                # Mask the loss to account for 'unmasked' tokens (neq to mask token id).
                cat_accumulate += sign * ((beta_cat * cat_align_ratio) * mask)
                if sign > 0:
                    num_loss = num_learn.loss * omega_t_num
                    cat_loss = cat_learn.loss * omega_t_cat * lose_is_masked
                # TOD: Add retani
            # Compute the (sim)NPO/DPO alignment loss.

            return (
                -F.logsigmoid(num_accumulate) + num_loss * 0.2,
                -F.logsigmoid(cat_accumulate) + cat_loss * 0.2,
            )
        else:

            win_is_masked = (
                win_output.categorical_output.xt == self.diffusion.mask_index
            ).float()
            lose_is_masked = (
                lose_output.categorical_output.xt == self.diffusion.mask_index
            ).float()
            # Create cross-wise reference loss.
            num_ref_win, cat_ref_win = (
                reference_win_output.continuous_output.loss,
                reference_win_output.categorical_output.loss * win_is_masked,
            )
            num_ref_los, cat_ref_los = (
                refernece_lose_output.continuous_output.loss,
                refernece_lose_output.categorical_output.loss * lose_is_masked,
            )

            # TODO: Use lose mask, as the learner will otherwise learn to 'descent' normally https://wandb.ai/j-m-galjaard-delft-university-of-technology/feta/runs/etj3e0x3?nw=nwuserjmgaljaard
            # Unpack learners (MASKED)
            num_learn_win, cat_learn_win = (
                win_output.continuous_output.loss,
                win_output.categorical_output.loss * lose_is_masked,
            )
            # Unpack learners (UNMASKED)
            num_learn_los, cat_learn_los = (
                lose_output.continuous_output.loss,
                lose_output.categorical_output.loss * lose_is_masked,
            )

            # IDEA: 'Align' with unconditional when conditioned, 'Disaling' unconditional wiht 'Conditional'
            # PREFERENTIAL FIRST.
            num_pref_ratio = num_learn_los - num_ref_win
            cat_pref_ratio = cat_learn_los - cat_ref_win
            # DISPREFERENTIAL SECOND.
            num_dispref_ratio = num_learn_win - num_ref_los
            cat_dispref_ratio = cat_learn_win - cat_ref_los

            beta_num = self._update_fn_cat.get_beta(num_pref_ratio, is_numeric=True)
            beta_cat = self._update_fn_num.get_beta(cat_pref_ratio, is_numeric=False)

            # TODO: mask the
            #   Removed omega_t  from ratio, as it diverged? https://wandb.ai/j-m-galjaard-delft-university-of-technology/feta/runs/lnbp3w65?nw=nwuserjmgaljaard

            num_cross_dpo = (
                -F.logsigmoid(beta_num * (num_pref_ratio - num_dispref_ratio))
                + 0.1 * num_learn_win * omega_t_num
            )
            #
            cat_cross_dpo = (
                -F.logsigmoid(beta_cat * (cat_pref_ratio - cat_dispref_ratio))
                + 0.1 * cat_learn_win * omega_t_cat * lose_is_masked
            )

            return num_cross_dpo, cat_cross_dpo
