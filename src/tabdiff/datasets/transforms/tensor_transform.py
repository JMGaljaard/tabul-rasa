import abc

import omegaconf
import torch
from torch.utils.data import default_collate

from tabdiff.auto import TensorTransformRegistry

from typing import Any, Callable, Dict, List


class BaseTensorTransform(abc.ABC, metaclass=TensorTransformRegistry):
    """
    Feature BaseTensorTransform registry, to resolve functional transform strategies in a torch.DataLoader.
    """

    def __init__(
        self,
        transform_fn: Callable[[torch.FloatTensor], torch.FloatTensor],
        *args,
        collate_fn: Callable[[torch.FloatTensor], torch.FloatTensor] = default_collate,
        **kwargs,
    ):
        super().__init__()
        self.transform_fn = transform_fn
        self.collate_fn = collate_fn

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

    def __call__(
        self, input_tensor: torch.FloatTensor | Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        collate_obj = self.collate_fn(input_tensor)
        if isinstance(collate_obj, torch.Tensor):

            return_obj = self.transform_fn(collate_obj)
        elif isinstance(collate_obj, dict):
            sample = collate_obj["sample"]

            return_obj = {
                "sample": sample,
                "perturbed": self.transform_fn(sample.clone(), sample_index=-1)[0],
            }
        return return_obj

    @classmethod
    def from_config(
        cls,
        cfg: omegaconf.OmegaConf | omegaconf.DictConfig,
        *args,
        collate_fn: Callable[[List[Any]], torch.FloatTensor] = None,
        **kwargs,
    ):

        if not hasattr(cfg.dataset, "transforms"):
            return None

        handler_class = cfg.dataset.transforms.get("method", None)
        return cls.get_registry()[handler_class]._from_config(
            cfg,
            *args,
            collate_fn=collate_fn,
            **kwargs,
        )
