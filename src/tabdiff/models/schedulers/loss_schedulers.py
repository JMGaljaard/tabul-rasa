import abc
import logging
from typing import Optional

import omegaconf

from tabdiff.auto.scale_scheduler import ScaleScheduleRegistry


class ScaleSchedule(metaclass=ScaleScheduleRegistry):
    """
    Abstract base class for loss scale schedulers.
    """

    def __init__(self, loss_lambda: float):
        self._loss_lambda = loss_lambda

    @abc.abstractmethod
    def step(self, step=None) -> float:
        """
        Updates and returns the current loss scale value.

        Args:
            step (int, *, None): Override step, to get scale factor for a specific step.
        """
        pass

    @abc.abstractmethod
    def _from_config(
        self, config: omegaconf.DictConfig, *args, **kwargs
    ) -> "ScaleSchedule":
        """
        Concreate implementation to construct object.
        Args:
            config ():
            *args ():
            **kwargs ():

        Returns:

        """
        pass

    @classmethod
    def from_config(cls, config: omegaconf.DictConfig, *args, **kwargs):
        key = (
            config.weight_schedule
            if hasattr(config, "weight_schedule")
            else "ConstantSchedule"
        )
        return (
            cls.get_registry()
            .get(key)
            ._from_config(
                config,
                *args,
                **kwargs,
            )
        )


class ConstantSchedule(ScaleSchedule):

    @classmethod
    def _from_config(cls, config: omegaconf.DictConfig, *args, **kwargs):
        lambda_or_default = (
            config.loss_lambda if hasattr(config, "loss_lambda") else 1.0
        )
        return ConstantSchedule(
            loss_lambda=lambda_or_default,
        )

    def __init__(self, loss_lambda: float = 1.0, *args, **kwargs):
        super().__init__(loss_lambda)

    def step(self, step: Optional[int] = None) -> float:
        return 1.0


class AnnealSchedule(ScaleSchedule):

    @classmethod
    def _from_config(
        self, config: omegaconf.DictConfig, *args, total_steps: int = None, **kwargs
    ):
        lambda_or_default = (
            config.loss_lambda if hasattr(config, "loss_lambda") else 1.0
        )
        steps = total_steps
        steps = steps or (config.steps if hasattr(config, "steps") else None)
        return AnnealSchedule(
            loss_lambda=lambda_or_default,
            total_steps=steps,
        )

    def __init__(self, loss_lambda: float = 1.0, total_steps: int = None):
        super().__init__(loss_lambda)
        self.total_steps = total_steps
        self.current_step = 0

    def step(self, step: Optional[int] = None) -> float:
        if self.current_step >= self.total_steps:
            return 0.0
        if step and abs(self.current_step - step) > 1:
            logging.warning(
                f"Observed step that exceeds expected step, updating self to {step}. "
                f"This may happen if you restart training, seeing this warnign more than once"
                f" indicates an error."
            )
            self.current_step = step
        scale = self._loss_lambda * (1 - self.current_step / self.total_steps)
        self.current_step += 1
        return scale
