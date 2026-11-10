import abc


class AugmentRegistry(abc.ABCMeta):
    """
    Note, this registry differs from Transform, as `Transform`s map from the pd.DataFrame (real/categorical) to
    the diffusion model's input domain. Augmentations apply transformations within a model's input domain.

    Dataset Registry meta-class, to enable registry lookup from_config.
    """

    REGISTRY = {}

    def __init__(cls, name, bases, attrs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        super().__init__(name, bases, attrs)
        if not cls.__abstractmethods__:
            cls.REGISTRY[cls.__name__] = cls

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)
