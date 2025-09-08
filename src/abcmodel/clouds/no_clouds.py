from ..models import AbstractCloudModel, AbstractMixedLayerModel


class NoCloudModel(AbstractCloudModel):
    """
    No cloud is formed using this model.
    """

    def __init__(self):
        self.cc_frac = 0.0
        self.cc_mf = 0.0
        self.cc_qf = 0.0

    def run(self, mixed_layer: AbstractMixedLayerModel):
        """No calculations."""
        pass
