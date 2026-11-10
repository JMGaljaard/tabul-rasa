from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from omegaconf import omegaconf
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabdiff.datasets import AlignDataset, TabDiffDataset
from tabdiff.datasets.collator import align_collator
from tabdiff.datasets.processors import inverse_data_transform
from tabdiff.eval import NLLAverageRunner
from tabdiff.models import UnifiedCtimeDiffusion
from tabdiff.unlearn.eval.templates import ULMetricBase

import torch


def compute_truth_ratio(
    nll_truth: torch.FloatTensor,
    nll_perturb: torch.FloatTensor,
    reduce: bool = True,
):
    """
    Compute log truth ratio from NLLs.

    Args:
        nll_truth (torch.FloatTensor): Tensor of shape [B] of true (unperturbed) samples NLL / ELBO.
        nll_perturb: Tensor of shape [B, K] NLLs of K perturbations / ELBO thereof.
        reduce (bool, default=True): Flag to disable reduction, to get truth ratio per perturbed sample. Defaults to
            True.
    Returns:
        ratio: Tensor of shape [B, 1] truth ratio.
    """
    # Expand nll_truth for broadcasting if batched

    # Compute P(perturb(x_i) | C) / P(x_i | C)
    #               = P(perturb(x_i),c)*P(C)) / P(x_i, c)P(C)
    #               = P(perturb(x_i),c) / P(x_i, c)
    #               = exp(log(P(perturb(x_i),c) / P(x_i, c))
    #               = exp(log(P(perturb(x_i),c)) - log(P(x_i, c))
    #               = exp(log(P(perturb(x_i),c)) - log(P(x_i, c))
    #               = exp(- log(P(x_i, c) - (-1 * log(P(perturb(x_i),c))))
    #               = exp(nll_truth - nll_perturb)
    with torch.device(nll_truth.device):
        if len(nll_truth.shape) == 1:
            # Create batch dimension
            nll_truth = nll_truth.unsqueeze(1)
        # Broadcasts automatically
        diffs = nll_truth - nll_perturb  # [B, K]
        if reduce:
            # [B, 1], normalize in log-space, log(1) == 0 --> so can use subtraction
            log_ratio = torch.logsumexp(diffs, dim=-1) - torch.log(
                torch.tensor(nll_perturb.shape[-1], dtype=diffs.dtype)
            )
        else:
            # Return individual ratio's
            log_ratio = diffs
        return torch.exp(log_ratio), log_ratio


class TabNLLMetrics(ULMetricBase):
    """
    Inversion / NLL based metric computation.
    """

    @classmethod
    def from_config(
        cls,
        config: omegaconf.DictConfig,
        metric_list: List[str],
        *args,
        **kwargs,
    ):

        dataset_kwargs = {
            "dequant_dist": config.dataset.dequant_dist,
            "int_dequant_factor": config.dataset.int_dequant_factor,
        }
        batch_size = config.sample.batch_size
        return cls(
            *args,
            config=config,
            dataset_kwargs=dataset_kwargs,
            inversion_batch_size=batch_size,
            metric_list=metric_list,
            **kwargs,
        )

    def __init__(
        self,
        real_data_path,
        test_data_path,
        val_data_path,
        info,
        device,
        reference_model: Optional[UnifiedCtimeDiffusion] = None,
        unlearn_model: UnifiedCtimeDiffusion = None,
        inversion_batch_size: int = 4096,
        dataset_kwargs=None,
        config: omegaconf.DictConfig = None,
        metric_list: List[str] = None,
    ):
        """

        Args:
            real_data_path ():
            test_data_path ():
            val_data_path ():
            info ():
            device ():
            reference_model (UnifiedCtimeDiffusion, *, None): Reference model, for optinoal Metrics. Note this model
                assumes to not make use of learning/updating.
            unlearn_model (UnifiedCtimeDiffusion, *, None): Unlearn model, for reference metrics.
        """

        super().__init__(
            real_data_path,
            test_data_path,
            val_data_path,
            info,
            device,
            config=config,
            ul_config=config.unlearn,
            metric_list=metric_list,
        )
        # TODO: maybe more of a builder type appraoch?
        self._reference_model = reference_model
        self._learnable_model = unlearn_model
        self._dataset_kwargs = dataset_kwargs or dict()
        self._inversion_batch_size = inversion_batch_size

    @torch.no_grad()
    def evaluate_nll_ratio(
        self,
        syn_df: pd.DataFrame,
        *args,
        perturbations: int = 2,
        # train_set: TabDiffDataset = None,
        # test_set: TabDiffDataset = None,
        # validation_set: Optional[TabDiffDataset] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError(
            "NLL ratio needs to be updated to construct its own TabDiffDataset."
        )

        # TODO: Implement loading of train / test /validation split

        # TODO: Determine whether augmentations shoudl be applied.
        # TODO: Create a dataloade.r
        averager = NLLAverageRunner(
            total_samples=len(syn_df),
        )

        # Re-use transformations to enable exact reconstruction of model input.
        (num_transform, int_transform, cat_transform) = fitted_transformations = (
            train_set.num_transform,
            train_set.int_transform,
            train_set.cat_transform,
        )
        (num_inverse, int_inverse, cat_inverse) = (
            transform.inverse_transform
            for transform in (num_transform, int_transform, cat_transform)
        )

        # This is really stupid, but so is the TabDiff code base.
        encoded_train_data: torch.FloatTensor = train_set.X
        decoded_train_frame = inverse_data_transform(
            train_set.info, encoded_train_data, cat_inverse, int_inverse, num_inverse
        )

        # TODO: How to do train/test split???
        frame_dict = {
            "train": decoded_train_frame,
        }

        align_dataset = AlignDataset.from_pandas(
            data=frame_dict,
            config=self._config,
            info=self.info,
            fitted_transformations=fitted_transformations,
        )
        align_loader = DataLoader(
            align_dataset,
            batch_size=self._config.sample.batch_size,
            shuffle=False,
            collate_fn=align_collator,
            num_workers=4,
        )
        # TODO: Implement geometric accumulator as well.
        # ALso maybe use a design pattern here instead. This is horrendous.

        for batch in tqdm(
            align_loader,
            desc="Iterating over align data",
            leave=False,
        ):
            # Compute truth (assumes only single truth existst)
            truth_input = batch["sample"].to(self.device)
            perturb_input = batch["perturbed"].to(self.device)
            batch_size = truth_input.size(0)

            batch_nll_original, batch_nll_elbo_original = (
                self._learnable_model.sample_reverse(truth_input)
            )

            # Compute distirbed nll
            if len(perturb_input.shape) == 2:
                # ONly a single augmentaition / perturbatino for the estimate.
                batch_nll_perturbed, batch_nll_eblo_perturbed = (
                    self._learnable_model.sample_reverse(perturb_input)
                )
                # No normaliziation needed.
                # TODO: Multiple perturbations??? / subsampling
            else:
                # Assumes tensor is returned
                batch_nll_perturbed = torch.zeros(
                    (batch_size, perturbations), dtype=torch.float32, device=self.device
                )
                assert batch_nll_perturbed.size(0) == perturb_input.size(1)
                for perturbation in range(perturbations):
                    perturbation_nll = self._learnable_model.sample_reverse(
                        perturb_input[perturbation]
                    )
                    batch_nll_perturbed[perturbation, :] = perturbation_nll

            batch_truth_ratios, batch_log_truth_ratios = compute_truth_ratio(
                nll_truth=batch_nll_original,
                nll_perturb=batch_nll_perturbed,
                reduce=True,
            )
            # Track Truth ratio, truth_nll (Bits per Column), pertrub_nll (Bits per Column)
            averager.update(
                {
                    "truth_ratio": batch_truth_ratios,
                    "truth_nll": batch_truth_ratios,
                    "truth_elbo": batch_nll_elbo_original,
                    "perturb_nll": batch_nll_perturbed,
                    "perturb_elbo": batch_nll_eblo_perturbed,
                }
            )

        # Collect statistics from averager.
        arith_means, geom_means, medians = averager.collect()
        # All results are important, no 'extra' results
        return {
            **arith_means,
            **geom_means,
            **medians,
        }, {}
