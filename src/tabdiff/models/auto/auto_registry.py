import abc


class TrainerRegistry(abc.ABCMeta):

    TRAINER_REGISTRY = {}

    def __init__(cls, name, bases, attrs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        super().__init__(name, bases, attrs)
        if not cls.__abstractmethods__:
            cls.TRAINER_REGISTRY[cls.__name__] = cls

    @classmethod
    def get_registry(cls):
        return dict(cls.TRAINER_REGISTRY)


class ModelRegistry(abc.ABCMeta):

    MODEL_REGISTRY = {}

    def __init__(cls, name, bases, attrs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        super().__init__(name, bases, attrs)
        if not cls.__abstractmethods__:
            cls.MODEL_REGISTRY[cls.__name__] = cls

    @classmethod
    def get_registry(cls):
        return dict(cls.MODEL_REGISTRY)
