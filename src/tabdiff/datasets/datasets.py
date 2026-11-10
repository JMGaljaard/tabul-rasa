import abc
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import omegaconf
import pandas as pd
import torch
from sklearn.base import TransformerMixin
from torch.utils.data import Dataset

from tabdiff.auto import DatasetRegistry
from tabdiff.datasets.processors import preprocess
from tabdiff.utils import Dequantizer, Transformations, remove_metadata_column


class TabDiffDatasetBase(Dataset, abc.ABC, metaclass=DatasetRegistry):

    @classmethod
    @abc.abstractmethod
    def _from_config(cls, config, *args, **kwargs):
        raise NotImplementedError("Can't call abstrac class method.")

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):

        if not hasattr(cfg, "dataset"):
            # Backwards compatibility
            return cls._from_config(cfg, *args, **kwargs)
        handler_class = cfg.dataset.get("handler", None)
        return cls.get_registry()[handler_class]._from_config(cfg, *args, **kwargs)


class TabularDataset(TabDiffDatasetBase):
    @classmethod
    def _from_config(cls, config, *args, **kwargs):
        return cls._from_config(*args, **kwargs)

    def __init__(self, X_num, X_cat):
        self.X_num = X_num
        self.X_cat = X_cat

    def __getitem__(self, index):
        this_num = self.X_num[index]
        this_cat = self.X_cat[index]

        sample = (this_num, this_cat)

        return sample

    def __len__(self):
        return self.X_num.shape[0]


class TabDiffDataset(TabDiffDatasetBase):

    @classmethod
    def from_pandas(
        cls,
        data: pd.DataFrame,
        info: Dict[str, Any],
        int_dequant_factor=None,
        dequant_dist=None,
        transforms: Transformations = None,
        fitted_transforms: Optional[
            Tuple[
                Optional[TransformerMixin],
                Optional[Dequantizer],
                Optional[TransformerMixin],
            ]
        ] = None,
    ):
        col_to_remove = None

        # Fitted-transforms is now a list
        # ( 1,          2,          3,              )
        (X_num, X_cat, categories, d_numerical, *fitted_transforms) = preprocess(
            Path("data") / info["name"],
            y_only=False,
            task_type=info["task_type"],
            inverse=True,
            info=info,
            transforms=transforms,
            fitted_transformations=fitted_transforms,
            data=None,
        )
        return cls(
            dataname=info.get("name", None),
            data_dir=None,
            X_num=X_num,
            X_cat=X_cat,
            categories=categories,
            d_numerical=d_numerical,
            info=info,
            int_dequant_factor=int_dequant_factor,
            dequant_dist=dequant_dist,
            col_to_remove=col_to_remove,
            transforms=transforms,
            fitted_transforms=fitted_transforms,
        )

    @classmethod
    def _from_config(
        cls,
        config,
        *args,
        fitted_transforms: Optional[
            Tuple[
                Optional[TransformerMixin],
                Optional[Dequantizer],
                Optional[TransformerMixin],
            ]
        ] = None,
        **kwargs,
    ):
        data_config = config
        if hasattr(config, "dataset"):
            # Support backwards
            data_config = config.dataset

        dataname = data_config.name
        dataset_dir = Path(kwargs.get("data_dir", "./data")) / dataname
        info_path = dataset_dir / "info.json"
        with open(info_path) as f:
            info = json.load(f)

        # TODO: Pre-processing configuration must be performed here/
        col_to_remove = None
        if (
            hasattr(data_config, "preprocessing")
            and data_config.preprocessing is not None
        ):
            col_to_remove = data_config.preprocessing.columns

        # Here, we update the meta-data of the info object,a
        if col_to_remove is None:
            info = info
        else:
            new_info = deepcopy(info)
            for col in sorted(col_to_remove, reverse=True):
                new_info = remove_metadata_column(new_info, col_idx_to_remove=col)
            info = new_info

        # Encoding transformations from config object.
        transforms = Transformations(
            **omegaconf.OmegaConf.to_container(
                data_config.transformations,
                resolve=True,
            )
        )

        # Fitted-transforms is now a list
        # ( 1,          2,          3,              )
        (X_num, X_cat, categories, d_numerical, *fitted_transforms) = preprocess(
            dataset_dir,
            y_only=False,
            task_type=info["task_type"],
            inverse=True,
            info=info,
            transforms=transforms,
            fitted_transformations=fitted_transforms,
            data=None,
        )

        return cls(
            *args,
            X_num=X_num,
            X_cat=X_cat,
            categories=categories,
            d_numerical=d_numerical,
            dataname=dataname,
            data_dir=dataset_dir,
            info=info,
            transforms=transforms,
            fitted_transforms=fitted_transforms,
            **kwargs,
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
        is_train=True,
        is_val=False,
        y_only=False,
        transforms: Transformations = None,
        fitted_transforms: Optional[
            Tuple[
                Optional[TransformerMixin],
                Optional[Dequantizer],
                Optional[TransformerMixin],
            ]
        ] = None,
        data: pd.DataFrame = None,
        **kwargs,
    ):
        """

        Args:
            dataname ():
            data_dir ():
            info ():
            is_train ():
            is_val ():
            y_only ():
            dequant_dist ():
            int_dequant_factor ():
            fitted_transforms (optional): Pre-fitted transforms, in case dataset is loaded for a pre-trained model.
            col_to_remove ():
            transforms ():
            data (): Pandas DataFrame containing generated data. N.B., when provided all other options will be ignored.
                Intended to be used for generated datasets, that do not 'exist' on disk.
        """
        self.dataname = dataname
        self.data_dir = data_dir

        self.info = info
        # TODO: Fix validation sometime
        self.is_train = is_train
        self.is_val = is_val
        assert not (
            self.is_train and self.is_val
        ), "Requested combination of dataset splits does not exist"
        # The transformations may be updated?
        self.transforms = transforms

        categories = np.array(categories)

        X_train_num, _ = X_num
        X_train_cat, _ = X_cat

        X_train_num, X_test_num = X_num
        X_train_cat, X_test_cat = X_cat

        X_train_num, X_test_num = (
            torch.tensor(X_train_num).float(),
            torch.tensor(X_test_num).float(),
        )
        X_train_cat, X_test_cat = torch.tensor(X_train_cat), torch.tensor(X_test_cat)

        self.X = (
            torch.cat((X_train_num, X_train_cat), dim=1)
            if is_train
            else torch.cat((X_test_num, X_test_cat), dim=1)
        )
        self.X = self.X.float()

        [num_transform, int_transform, cat_transform] = fitted_transforms
        self.num_transform = num_transform
        self.int_transform = int_transform
        self.cat_transform = cat_transform
        self.d_numerical = d_numerical
        self.categories = categories

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.X.shape[0]
