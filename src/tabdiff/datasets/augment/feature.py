import abc
from functools import reduce

import omegaconf
import torch

from tabdiff.auto import AugmentRegistry

from typing import Callable, List, Tuple


def _apply(accum, fn):
    return fn(*accum)


class BaseFeatureAugment(abc.ABC, metaclass=AugmentRegistry):
    """
    Feature Augmentation registry, to resolve augmentation strategies in a model's input domain
    augmentation after Transform.
    """

    def __init__(
        self,
        transformations: List[
            Callable[
                [torch.FloatTensor, torch.LongTensor],
                Tuple[torch.FloatTensor, torch.LongTensor],
            ]
        ],
        *args,
        **kwargs,
    ):
        self.transformations = transformations

    def __call__(
        self, sample: torch.FloatTensor, index: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        Apply Augment transformations on pre-encoded dataste.
        Args:
            sample ():
            index ():

        Returns:

        """
        # Apply self-applier to original input, i.e. recursively apply transformations.
        return reduce(_apply, self.transformations, (sample, index))[0]

    @classmethod
    @abc.abstractmethod
    def _from_config(
        cls,
        config: omegaconf.DictConfig,
        dataset: torch.FloatTensor,
        *args,
        **kwargs,
    ):
        raise NotImplementedError("Can't call abstract class method.")

    @classmethod
    def from_config(
        cls,
        cfg: omegaconf.OmegaConf | omegaconf.DictConfig,
        encoded_data: torch.FloatTensor,
        *args,
        **kwargs,
    ):

        if not hasattr(cfg.dataset, "augment"):
            # Backwards compatibility, use provided __class__ to resolve
            return cls._from_config(cfg, encoded_data, *args, **kwargs)
        # Resolve from config
        # TODO: Figure out how to do this from an `eval` or `...` config.
        handler_class = cfg.dataset.get("augment", None)
        return cls.get_registry()[handler_class]._from_config(
            cfg, encoded_data, *args, **kwargs
        )
