import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import pandas as pd
import omegaconf
from tabdiff.datasets.datasets import TabDiffDataset
from tabdiff.datasets.processors.postprocessors import inverse_data_transform


class TabMetricBase(ABC):
    """
    Base class for Tabular Metrics.
    """

    def __init__(
        self,
        real_data_path,
        test_data_path,
        val_data_path,
        info,
        device,
        config: omegaconf.DictConfig,
    ):
        self._config = config
        self.real_data_path = real_data_path
        self.test_data_path = test_data_path
        self.val_data_path = val_data_path

        self.info = info
        self.device = device

    def get_data(self, split: str = "real") -> pd.DataFrame:
        """
        Reconstructs a datasplit.
        Args:
            split (str, 'real'): Split of data to reconstruct

        Returns:

        """
        is_train = split == "real"
        is_val = split == "val"
        if split not in {"real", "val", "test"}:
            logging.warning("Default to train split")
        dataset = TabDiffDataset.from_config(
            self._config.dataset,
            y_only=False,
            is_train=is_train,
            is_val=is_val,
        )
        df = inverse_data_transform(
            self.info,
            dataset.X,
            dataset.cat_transform.inverse_transform,
            dataset.int_transform.inverse_transform,
            dataset.num_transform.inverse_transform,
        )
        return df

    @abstractmethod
    def evaluate(
        self,
        syn_data,
        *args,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Method to evalaute generated/synthetic data post-processing
        Args:
            syn_data ():
            args (tuple): Additional args required by the callee.
            kwargs (dict): Additional kwargs required by the callee.

        Returns:
            Dict[str, Any]: Result dictionary for provided data.
            Dict[str, Any]: Additional result dictionary for provided data.
        """
        raise NotImplementedError
