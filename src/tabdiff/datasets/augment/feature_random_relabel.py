import logging
from functools import reduce

import numpy as np
import omegaconf
import torch

from .feature import BaseFeatureAugment
from typing import Optional, Union, Any, Dict, Callable, Tuple
from tabdiff.utils import Dataset, load_dataset_info


def idk_transform(
    cardinalities: np.ndarray,
    indices: np.ndarray,
) -> Callable[
    [torch.FloatTensor, torch.LongTensor], Tuple[torch.FloatTensor, torch.LongTensor]
]:
    """
    Targeted Model input domain transformation for features that lie in the categorical domain to UNKNOWN tokens.
    Args:
        cardinalities (np.ndarray): Cardinalities of the targeted columns.
        indices (torch.LongTensor): Encoded data's indices to transform (noise).

    Returns:
        callable: Function implementing 'UNK' token relabeling.
    """

    def tranform_fn(
        sample: torch.FloatTensor, index: torch.LongTensor = None
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        nonlocal cardinalities, indices
        if len(sample.shape) > 0:
            # Assumes that highest cardinality is the IDK token...
            # Note this is not really expected to be used
            sample[:, indices] = cardinalities[None, :]
        else:
            #
            sample[indices] = cardinalities

        return sample, index

    return tranform_fn


def model_domain_categorical_transform(
    cardinalities: np.ndarray,
    indices: np.ndarray,
) -> Callable[
    [torch.FloatTensor, torch.LongTensor], Tuple[torch.FloatTensor, torch.LongTensor]
]:
    """
    Targeted Model input domain transformation for features that lie in the categorical domain to random tokens.
    Args:
        cardinalities (np.ndarray): Cardinalities of the targeted columns.
        indices (torch.LongTensor): Encoded data's indices to transform (noise)

    Returns:
        callable: Function implementing random relabeling strategy.
    """
    generator = np.random.default_rng()
    torch_cardinalities = torch.from_numpy(cardinalities)
    torch_indices = torch.from_numpy(indices)

    def transform_fn(
        sample: torch.FloatTensor, index: torch.LongTensor = None
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        nonlocal generator, torch_indices, torch_cardinalities
        original_shape = sample.shape
        if len(sample.shape) == 1:
            # Make sure everything can be done as a batched operator.
            sample = sample.reshape((1, -1))
        resampled = sample.clone().detach()
        batch_size = sample.shape[0]
        # can't do batched integer resampling in torch.
        # See: https://github.com/pytorch/pytorch/issues/89438
        for cardinality_index, index in enumerate(torch_indices):
            # Random offset of atleast 1

            # Maybe the bug lies here, we relabel also to the UNK token.. No?
            #   Don't think so, because we do filter out tokens that have the UNK index.
            #   It may cause it to be excluded from the overal unlearning process, especially in binary case.
            #

            # With multiple perturbation requests... (binary class), the tester
            # Should be aware of this / smarter handling.
            offset = torch.randint(
                size=(batch_size,),
                low=1,
                high=torch_cardinalities[cardinality_index].item(),
            )
            # Update samples with wrapped around
            resampled[:, index] = (
                (sample[:, index].to(torch.long) + offset.reshape(-1))
                % torch_cardinalities[cardinality_index]
            ).to(sample.dtype)
        return resampled.reshape(original_shape), index

    return transform_fn


def model_domain_numerical_transform(
    margin: float = 1e-4,
    indices: torch.Tensor = None,
) -> Callable[
    [torch.FloatTensor, torch.LongTensor], Tuple[torch.FloatTensor, torch.LongTensor]
]:
    """
    Targeted Model input domain transformation for features that lie in the numerical domain.
    Args:
        margin (float): Margin that should be achieved by re-sampling
        indices (torch.LongTensor): Encoded data's indices to transform (noise)

    Returns:
        callable: Function implementing the (random) augmentation strategy.
    """
    logging.warning("Numerical transform assumes Gaussian prior...")
    generator = torch.Generator()
    assert indices is not None, "Indices are required for numerical perturbation"
    torch_indices = torch.from_numpy(indices).to(torch.long)

    def transform_fn(
        sample: torch.FloatTensor, index: torch.LongTensor = None
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        nonlocal margin, generator, margin, torch_indices
        original_shape = sample.shape
        if len(sample.shape) == 1:
            # Make sure everything can be done as a batched operator.
            sample = sample.reshape((1, -1))
        resampled = sample.clone().detach()

        # Within a reasonable number of attempts the margin should be achieved.
        attempts = 3
        while attempts > 0:
            # Assume torch_indices is a 1D tensor of indices
            resampled_vals = resampled[:, torch_indices]
            sample_vals = sample[:, torch_indices]

            # Compute where the difference is too small
            too_close_mask = (
                torch.abs(resampled_vals - sample_vals) < margin
            )  # shape: (B, N)

            # Count how many total need resampling
            resample_count = too_close_mask.sum()
            if resample_count < 1:
                break
            # Get the indices to replace
            batch_indices, index_in_tensor = torch.nonzero(
                too_close_mask, as_tuple=True
            )
            # Resample new values
            new_values = torch.randn(
                size=(resample_count,),
                generator=generator,
                device=resampled.device,
            )
            # Construct reverse index
            true_col_indices = torch_indices[
                index_in_tensor
            ]  # Actual column indices in resampled
            # Use scatter to update only the specific values
            resampled[batch_indices, true_col_indices] = new_values
            attempts -= 1
        return resampled.reshape(original_shape), index

    return transform_fn


class RandomPerturbAugment(BaseFeatureAugment):
    """
    Implements 'dynamic' (random per call) transforms on pre-encoded data (diffusion model domain).



    Notes:
        This only exists, because the data-handling by tabdiff is rather inane.
        Assumes that the data index (requested atleast) is available.
    """

    @classmethod
    def _from_config(
        cls,
        config: omegaconf.OmegaConf | omegaconf.DictConfig,
        dataset: torch.FloatTensor,
        *args,
        info: Optional[Dict[str, Any]] = None,
        cat_cardinalities: Optional[np.ndarray[int]] = None,
        **kwargs,
    ):
        """

        Args:
            config (): Experiment configuration object.
            *args ():
            info (): Dataset meta-data.
            cat_cardinalities (): Cardinalities per columns.
            **kwargs ():

        Returns:
            RandomPerturbAugment instance configured with requested augmentation strategy.
        """
        # 1. Get dataset info to find mapping after pre-processing
        #   (Note, this is regarded)
        info = info or load_dataset_info(config.dataset.name)
        # Get lut offset in case there is a numerical column/s
        data_to_representation_lut: Optional[Dict[Union[str, int], Union[str, int]]] = (
            info.get("idx_mapping", None)
        )
        assert (
            data_to_representation_lut is not None
        ), "Are you refactoring the processing pipeline? Metadata object is broken"
        numerical_offset = len(info.get("num_col_idx", []))
        # Determine if margin is needed.
        output_dist_offset = info.get("task_type") != "regression"

        # 2. Get columns to 'remove'
        unlearn_config = config.unlearn
        target_columns = (
            unlearn_config.columns
            if hasattr(unlearn_config, "columns")
            else [unlearn_config.column]
        )
        target_dtypes = (
            unlearn_config.dtypes
            if hasattr(unlearn_config, "dtypes")
            else [unlearn_config.dtype]
        )

        num_indices, cat_indices, cat_cardinal = [], [], []

        for data_index, dtype in zip(target_columns, target_dtypes):
            encoded_index = data_to_representation_lut[data_index]
            match dtype:
                case "numerical":
                    num_indices.append(encoded_index)
                case "categorical":
                    # TODO: Figure out cardinality of dataset
                    # if is discrete output distribution,it is prepended (...., yes...)
                    corrected_tensor_index = (
                        int(encoded_index) - numerical_offset + output_dist_offset
                    )
                    cat_indices.append(int(encoded_index) + output_dist_offset)

                    cat_cardinal.append(cat_cardinalities[corrected_tensor_index])
                case _:
                    raise NotImplementedError(
                        f"Data of type: {dtype} is not supported, at index {data_index}."
                    )

        # Define transformations to perform on the encoded data-domain (diffusion model input)
        transforms = []
        if len(num_indices) > 0:
            transforms.append(
                model_domain_numerical_transform(
                    indices=np.array(num_indices),
                )
            )
        if len(cat_indices) > 0:
            transforms.append(
                model_domain_categorical_transform(
                    indices=np.array(cat_indices),
                    cardinalities=np.array(cat_cardinal),
                )
            )

        # Construct pipeline by applying transformations onto data.
        # Self-application of argument
        apply = lambda accum, fn: fn(*accum)
        # Apply self-applier to original input, i.e. recursively apply transformations.
        apply_transforms = lambda sample, index: reduce(
            apply, transforms, (sample, index)
        )

        return RandomPerturbAugment(
            apply_transforms,
            transforms,
        )
