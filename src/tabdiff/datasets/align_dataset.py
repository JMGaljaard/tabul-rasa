import logging

import numpy as np
import omegaconf
import torch
from sklearn.base import TransformerMixin

from .augment import BaseFeatureAugment
from .datasets import TabDiffDatasetBase
import pandas as pd

from typing import Callable, Dict, Any, List, Optional, Tuple, TypeVar
from pathlib import Path

from .processors import concat_encoded_data, preprocess
from tabdiff.utils import Dequantizer, Transformations, load_dataset_info

# Define framedict
FrameDict = TypeVar(
    "FrameDict",
    bound=Dict[str, pd.DataFrame],
)


class AlignDataset(TabDiffDatasetBase):

    @classmethod
    def from_pandas(
        cls,
        data: FrameDict,
        config: omegaconf.DictConfig,
        info: Dict[str, Any],
        fitted_transformations: Optional[Tuple[Transformations, ...]] = None,
    ):
        """
        Construct 'aligned' version of dataset, using loaded dataframe.

        Notes:
            This function is currently intended only for NLL computations, where random perturbations are requested
            to compute 'Truth' ratios.

        Args:
            data (dict): str->pd.DataFrame mapping for different splits of data.
            config (): Experimental configuration
            info (): Dataset transformations
            fitted_transformations (): Pre-fitted transformations, st. encodign of data is consistent
                with input domain of the model.

        Returns:
            AlignDataset: Align-dataste seeded with the provided training data.
        """
        # Data should already be removed if this is called

        col_to_remove = None
        data_name = None
        data_dir = None

        logging.warning(
            "No Transformations provided. New Transformations will be fitted on provided `data` frames."
        )
        transformations = Transformations(
            **omegaconf.OmegaConf.to_container(
                config.dataset.transformations, resolve=True
            )
        )
        # Fitted-transforms is now a list
        (X_num, X_cat, categories, d_numerical, *fitted_transformations) = preprocess(
            data_dir,
            y_only=False,
            task_type=info["task_type"],
            inverse=True,
            info=info,
            transforms=transformations,
            fitted_transformations=fitted_transformations,
            data=data,
            col_to_remove=col_to_remove,
        )

        X_num_split, _ = X_num
        X_cat_split, _ = X_cat

        # This is specific to the 'align' dataset
        align_perturb = BaseFeatureAugment.from_config(
            config,
            encoded_data=concat_encoded_data(
                X_cat_split,
                X_num_split,
            ).float(),
            cat_cardinalities=categories,
            info=info,
        )

        return AlignDataset(
            dataname=config.dataset.name,
            data_dir=Path(
                info.get(
                    "data_path", f"data/{config.dataset.name}/this_should_not_happen"
                )
            ).parent,
            X_num=X_num_split,
            X_cat=X_cat_split,
            categories=categories,
            d_numerical=d_numerical,
            info=info,
            is_train=True,
            is_val=False,
            is_test=False,
            transformations=transformations,
            fitted_transforms=fitted_transformations,
            perturb_fn=align_perturb,
        )

    @classmethod
    def _from_config(
        cls,
        config,
        *args,
        is_train: bool = True,
        is_test: bool = False,
        is_val: bool = False,
        fitted_transforms: Optional[
            Tuple[
                Optional[TransformerMixin],
                Optional[Dequantizer],
                Optional[TransformerMixin],
            ]
        ] = None,
        num_perturb_fn: Optional[Callable[[Any, ...], np.ndarray]] = None,
        cat_perturb_fn: Optional[Callable[[Any, ...], np.ndarray]] = None,
        **kwargs,
    ):
        assert hasattr(config, "dataset"), (
            "For align datasets, the whole config is required"
            " as the implementation................."
        )
        data_config = config.dataset
        dataname = data_config.name
        dataset_path = Path(kwargs.get("data_dir", "./data")) / dataname

        info = load_dataset_info(config.dataset.name)

        col_to_remove = None
        if (
            hasattr(data_config, "preprocessing")
            and data_config.preprocessing is not None
        ):

            col_to_remove = data_config.preprocessing.columns

        transforms = Transformations(
            **omegaconf.OmegaConf.to_container(
                data_config.transformations,
                resolve=True,
            )
        )
        (X_num, X_cat, categories, d_numerical, *fitted_transforms) = preprocess(
            dataset_path,
            y_only=False,
            task_type=info["task_type"],
            inverse=True,
            info=info,
            transforms=transforms,
            fitted_transformations=fitted_transforms,
            data=None,
            col_to_remove=col_to_remove,
        )

        X_num_train, X_num_test = X_num
        X_cat_train, X_cat_test = X_cat
        X_num_split = X_num_train if is_train else X_num_test
        X_cat_split = X_cat_train if is_train else X_cat_test

        # Lastly, we create a transform object that describes the transformations to be performanced
        align_perturb = None
        if config.dataset.augment is not None:
            align_perturb = BaseFeatureAugment.from_config(
                config,
                encoded_data=concat_encoded_data(X_cat_split, X_num_split).float(),
                cat_cardinalities=categories,
                info=info,
            )

        return AlignDataset(
            *args,
            dataname=config.dataset.name,
            data_dir=Path(
                info.get(
                    "data_path", f"data/{config.dataset.name}/this_should_not_happen"
                )
            ).parent,
            X_num=X_num_split,
            X_cat=X_cat_split,
            categories=categories,
            d_numerical=d_numerical,
            info=info,
            is_train=is_train,
            is_val=is_val,
            is_test=is_test,
            transformations=transforms,
            fitted_transforms=fitted_transforms,
            perturb_fn=align_perturb,
        )

    def __init__(
        self,
        dataname,
        data_dir,
        X_num: Optional[np.ndarray],
        X_cat: Optional[np.ndarray],
        categories: List[int],
        d_numerical: int,
        info,
        *args,
        is_train=True,
        is_val=False,
        is_test=False,
        y_only=False,
        transformations: Transformations = None,
        fitted_transforms: Optional[
            Tuple[
                Optional[Transformations],
                Optional[Transformations],
                Optional[Transformations],
            ]
        ] = None,
        perturb_fn: Optional[
            Callable[
                [torch.FloatTensor, torch.LongTensor],
                Tuple[torch.FloatTensor, torch.LongTensor],
            ]
        ] = None,
        **kwargs,
    ):

        self.dataname = dataname
        self.data_dir = data_dir

        self.info = info
        # TODO: Fix validation sometime
        self.is_train = is_train
        self.is_val = is_val
        self.is_test = is_test
        assert not (
            self.is_train and self.is_val
        ), "Requested combination of dataset splits does not exist"
        self.transforms = transformations
        [num_transform, int_transform, cat_transform] = fitted_transforms
        self.num_transform = num_transform
        self.int_transform = int_transform
        self.cat_transform = cat_transform

        self.X = concat_encoded_data(X_cat, X_num)

        self.d_numerical = d_numerical
        self.categories = categories

        self.perturb_fn = perturb_fn

    def __getitem__(self, index):
        original_sample = self.X[index]
        if self.perturb_fn is not None:
            # Now provide index to ...
            return {
                "sample": original_sample,
                "perturbed": self.perturb_fn(original_sample, index=index),
            }
        else:
            # In this case, the perturb function should be implemented by the collation function of the dataloader
            return {
                "sample": original_sample,
                "perturbed": original_sample,
            }

    def __len__(self):
        return self.X.shape[0]
