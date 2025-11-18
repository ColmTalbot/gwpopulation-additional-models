from gwpopulation.models.redshift import PowerLawRedshift
from gwpopulation.models.spin import (
    GaussianChiEffChiP as GaussianEffectiveSpins
)

from .mixture import MixtureOfPowerLawsAndGaussians
from .spin import (
    truncated_normal_spin_magnitude_iid,
    truncated_normal_spin_magnitude_independent,
)

__all__ = [
    "BrokenPowerLawTwoPeak",
    "GaussianEffectiveSpins",
    "PowerLawRedshift",
    "truncated_normal_spin_magnitude_iid",
    "truncated_normal_spin_magnitude_independent",
]


class BrokenPowerLawTwoPeak(MixtureOfPowerLawsAndGaussians):

    def __init__(self, gaussian_maximum: float = 100):
        super().__init__(n_powerlaws=2, n_gaussians=2, gaussian_maximum=100)
