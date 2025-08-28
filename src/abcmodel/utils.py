import numpy as np


def get_esat(T):
    return 0.611e3 * np.exp(17.2694 * (T - 273.16) / (T - 35.86))


def get_qsat(T, p):
    return 0.622 * get_esat(T) / p


def get_psim(zeta):
    if zeta <= 0:
        x = (1.0 - 16.0 * zeta) ** (0.25)
        psim = (
            3.14159265 / 2.0
            - 2.0 * np.arctan(x)
            + np.log((1.0 + x) ** 2.0 * (1.0 + x**2.0) / 8.0)
        )
        # x     = (1. + 3.6 * abs(zeta) ** (2./3.)) ** (-0.5)
        # psim = 3. * np.log( (1. + 1. / x) / 2.)
    else:
        psim = (
            -2.0 / 3.0 * (zeta - 5.0 / 0.35) * np.exp(-0.35 * zeta)
            - zeta
            - (10.0 / 3.0) / 0.35
        )
    return psim


def get_psih(zeta):
    if zeta <= 0:
        x = (1.0 - 16.0 * zeta) ** (0.25)
        psih = 2.0 * np.log((1.0 + x * x) / 2.0)
        # x     = (1. + 7.9 * abs(zeta) ** (2./3.)) ** (-0.5)
        # psih  = 3. * np.log( (1. + 1. / x) / 2.)
    else:
        psih = (
            -2.0 / 3.0 * (zeta - 5.0 / 0.35) * np.exp(-0.35 * zeta)
            - (1.0 + (2.0 / 3.0) * zeta) ** (1.5)
            - (10.0 / 3.0) / 0.35
            + 1.0
        )
    return psih


class PhysicalConstants:
    """Container for physical constants used throughout the model."""

    def __init__(self):
        # 1. thermodynamic constants:
        # heat of vaporization [J kg-1]
        self.lv = 2.5e6
        # specific heat of dry air [J kg-1 K-1]
        self.cp = 1005.0
        # density of air [kg m-3]
        self.rho = 1.2
        # gravity acceleration [m s-2]
        self.g = 9.81
        # gas constant for dry air [J kg-1 K-1]
        self.rd = 287.0
        # gas constant for moist air [J kg-1 K-1]
        self.rv = 461.5
        # density of water [kg m-3]
        self.rhow = 1000.0

        # 2. physical constants:
        # von Karman constant [-]
        self.k = 0.4
        # Boltzmann constant [-]
        self.bolz = 5.67e-8
        # solar constant [W m-2]
        self.solar_in = 1368.0

        # 3. molecular weights:
        # molecular weight CO2 [g mol-1]
        self.mco2 = 44.0
        # molecular weight air [g mol-1]
        self.mair = 28.9
        # ratio molecular viscosity water to carbon dioxide
        self.nuco2q = 1.6


def get_ribtol(Rib, zsl, z0m, z0h):
    if Rib > 0.0:
        L = 1.0
        L0 = 2.0
    else:
        L = -1.0
        L0 = -2.0

    while abs(L - L0) > 0.001:
        L0 = L
        fx = (
            Rib
            - zsl
            / L
            * (np.log(zsl / z0h) - get_psih(zsl / L) + get_psih(z0h / L))
            / (np.log(zsl / z0m) - get_psim(zsl / L) + get_psim(z0m / L)) ** 2.0
        )
        Lstart = L - 0.001 * L
        Lend = L + 0.001 * L
        fxdif = (
            (
                -zsl
                / Lstart
                * (np.log(zsl / z0h) - get_psih(zsl / Lstart) + get_psih(z0h / Lstart))
                / (np.log(zsl / z0m) - get_psim(zsl / Lstart) + get_psim(z0m / Lstart))
                ** 2.0
            )
            - (
                -zsl
                / Lend
                * (np.log(zsl / z0h) - get_psih(zsl / Lend) + get_psih(z0h / Lend))
                / (np.log(zsl / z0m) - get_psim(zsl / Lend) + get_psim(z0m / Lend))
                ** 2.0
            )
        ) / (Lstart - Lend)
        L = L - fx / fxdif

        if abs(L) > 1e15:
            break

    return L
