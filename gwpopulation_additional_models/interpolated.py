from gwpopulation.models.interped import InterpolatedNoBaseModelIdentical
from gwpopulation.models.redshift import _Redshift
from gwpopulation.utils import xp


class InterpolatedRedshift(_Redshift, InterpolatedNoBaseModelIdentical):
    r"""
    Exponentiated cubic spline model for the redshift evolution of the merger
    rate.

    ... math::

        psi(z) = exp(spl(z))

    """

    def __init__(
        self,
        zmax,
        nodes=10,
        kind="cubic",
        log_nodes=False,
        regularize=False,
    ):
        """ """
        _Redshift.__init__(self, z_max=zmax)
        InterpolatedNoBaseModelIdentical.__init__(
            self,
            parameters=["redshift"],
            minimum=0,
            maximum=zmax,
            nodes=nodes,
            kind=kind,
            log_nodes=log_nodes,
            regularize=regularize,
        )
        self._xs = self.zs_

    @property
    def variable_names(self):
        return InterpolatedNoBaseModelIdentical.variable_names.fget(self)

    def normalisation(self, parameters):
        r"""
        Compute the normalization or differential spacetime volume.
        .. math::
            \mathcal{V} = \int dz \frac{1}{1+z} \frac{dVc}{dz} \psi(z|\Lambda)
        Parameters
        ----------
        parameters: dict
            Dictionary of parameters
        Returns
        -------
        (float, array-like): Total spacetime volume
        """
        f_splines = xp.array([parameters[key] for key in self.fkeys])
        x_splines = xp.array([parameters[key] for key in self.xkeys])

        psi_of_z = xp.exp(self._norm_spline(y=f_splines))
        psi_of_z *= (self._xs >= x_splines[0]) & (self._xs <= x_splines[-1])
        norm = xp.trapz(psi_of_z * self.dvc_dz / (1 + self.zs), self.zs)
        return norm

    def psi_of_z(self, redshift, **parameters):
        self.infer_n_nodes(**parameters)

        f_splines = xp.array([parameters[key] for key in self.fkeys])
        x_splines = xp.array([parameters[key] for key in self.xkeys])

        return self.p_x_unnormed(
            dict(redshift=redshift),
            "redshift",
            x_splines=x_splines,
            f_splines=f_splines,
            **parameters
        )
