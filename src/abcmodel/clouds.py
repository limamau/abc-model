import numpy as np

from .components import AbstractCloudModel, AbstractMixedLayerModel
from .utils import get_qsat


class NoCloudModel(AbstractCloudModel):
    def __init__(self):
        # cloud core fraction [-]
        self.cc_frac = 0.0
        # cloud core mass flux [m s-1]
        self.cc_mf = 0.0
        # cloud core moisture flux [kg kg-1 m s-1]
        self.cc_qf = 0.0

    def run(self, mixed_layer: AbstractMixedLayerModel):
        pass


class StandardCumulusModel(AbstractCloudModel):
    # limamau: should this really be intialized like that?
    # or would we rather actually use args?
    def __init__(self):
        # cloud core fraction [-]
        self.cc_frac = 0.0
        # cloud core mass flux [m s-1]
        self.cc_mf = 0.0
        # cloud core moisture flux [kg kg-1 m s-1]
        self.cc_qf = 0.0

    def run(self, mixed_layer: AbstractMixedLayerModel):
        # calculate mixed-layer top relative humidity variance (Neggers et. al 2006/7)
        if mixed_layer.wthetav > 0.0:
            mixed_layer.q2_h = (
                -(mixed_layer.wqe + self.cc_qf)
                * mixed_layer.dq
                * mixed_layer.abl_height
                / (mixed_layer.dz_h * mixed_layer.wstar)
            )
            mixed_layer.top_CO22 = (
                -(mixed_layer.wCO2e + mixed_layer.wCO2M)
                * mixed_layer.dCO2
                * mixed_layer.abl_height
                / (mixed_layer.dz_h * mixed_layer.wstar)
            )
        else:
            mixed_layer.q2_h = 0.0
            mixed_layer.top_CO22 = 0.0

        # calculate cloud core fraction (ac), mass flux (M) and moisture flux (wqM)
        self.cc_frac = max(
            0.0,
            0.5
            + (
                0.36
                * np.arctan(
                    1.55
                    * (
                        (mixed_layer.q - get_qsat(mixed_layer.top_T, mixed_layer.top_p))
                        / mixed_layer.q2_h**0.5
                    )
                )
            ),
        )
        self.cc_mf = self.cc_frac * mixed_layer.wstar
        self.cc_qf = self.cc_mf * mixed_layer.q2_h**0.5

        # only calculate CO2 mass-flux if mixed-layer top jump is negative
        if mixed_layer.dCO2 < 0:
            mixed_layer.wCO2M = self.cc_mf * mixed_layer.top_CO22**0.5
        else:
            mixed_layer.wCO2M = 0.0
