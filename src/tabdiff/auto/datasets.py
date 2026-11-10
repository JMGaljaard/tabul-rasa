import abc


class DatasetRegistry(abc.ABCMeta):
    """
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
