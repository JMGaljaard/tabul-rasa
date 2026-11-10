from tabdiff.eval import TabMetrics


class ULMetricBase(TabMetrics):
    """
    Base class to enable ext
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        maybe_conf = kwargs.get("config", None)
        self.ul_metric = None if maybe_conf is None else maybe_conf.unlearn
