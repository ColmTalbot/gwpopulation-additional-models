from gwpopulation.utils import powerlaw, truncnorm, xp


class MixtureOfPowerLawsAndGaussians:
    """
    A one-dimensional model for an arbitrary combination of power-law and
    Gaussian components.

    Each power-law component requires two variables :code:`alpha, break`.
    The :code:`alpha` are the spectral indices and the :code:`break` are
    the positions of the break points. Note that there is one less
    :code:`break` variables than :code:`alpha`.

    Each Gaussian component requires three variables :code:`weight,mean,sigma`.
    The weights are capped to sum to at most 1.

    There are two additional variables, :code:`minimum` (the global minimum)
    and :code:`maximum` (the maximum of the power-law component).
    """

    def __init__(
        self,
        n_powerlaws: int,
        n_gaussians: int,
        gaussian_maximum: float = 100,
        name: str = "",
        key_mapping: dict = None,
    ):
        """
        Parameters
        ==========
        n_powerlaws: int
            The number of power-law components
        n_gaussians: int
            The number of Gaussian components
        gaussian_maximum: float
            The cutoff value for the maximum of the Gaussian components
        name: str
            The name of the parameter being fit, e.g., :code:`"mass"`. This
            should be a prefix for all parameters of the model, e.g.,
            :code:`"mass_alpha_1"`.
        key_mapping: dict
            A mapping between the standardized variable names and
            user-specified.
        """
        self.n_powerlaws = n_powerlaws
        self.n_gaussians = n_gaussians
        self.gaussian_maximum = gaussian_maximum
        if name == "" or name.endswith("_"):
            self.name = name
        else:
            self.name = name
        if key_mapping is None:
            self.key_mapping = dict()
        else:
            self.key_mapping = key_mapping

    def replace_key(self, key):
        base = key.split("_")[0]
        return key.replace(self.key_mapping.get(base, base))

    @property
    def variable_names(self):
        variables = list()
        alpha = self.replace_key("alpha")
        break_ = self.replace_key("break")
        weight = self.replace_key("weight")
        mean = self.replace_key("mean")
        sigma = self.replace_key("sigma")
        if self.n_powerlaws > 1:
            variables += [f"{alpha}_{ii + 1}" for ii in range(self.n_powerlaws)]
            variables += [f"{break_}_{ii}" for ii in range(1, self.n_powerlaws)]
        elif self.n_powerlaws == 1:
            variables += [alpha]
        if self.n_gaussians > 1:
            variables += [f"{weight}_{ii + 1}" for ii in range(self.n_gaussians)]
            variables += [f"{mean}_{ii + 1}" for ii in range(self.n_gaussians)]
            variables += [f"{sigma}_{ii + 1}" for ii in range(self.n_gaussians)]
        elif self.n_gaussians == 1:
            variables += [weight, mean, sigma]
        variables += [self.replace_key(key) for key in ["minimum", "maximum"]]
        variables = [f"{self.name}{variable}" for variable in variables]
        return variables

    def gaussian_component(self, data, **kwds):
        if self.n_gaussians == 0:
            return 0
        elif self.n_gaussians == 1:
            return truncnorm(
                data,
                mu=kwds["mean"],
                sigma=kwds["sigma"],
                high=self.gaussian_maximum,
                low=kwds["minimum"],
            )
        else:
            return xp.sum(
                xp.array(
                    [
                        kwds[f"weight_{ii + 1}"]
                        * truncnorm(
                            data,
                            mu=kwds[f"mean_{ii + 1}"],
                            sigma=kwds[f"sigma_{ii + 1}"],
                            high=self.gaussian_maximum,
                            low=kwds["minimum"],
                        )
                        for ii in range(self.n_gaussians)
                    ]
                ),
                axis=0,
            ) / xp.sum(
                xp.array([kwds[f"weight_{ii + 1}"] for ii in range(self.n_gaussians)])
            )

    def powerlaw_component(self, data, **kwds):
        if self.n_powerlaws == 0:
            return 0
        elif self.n_powerlaws == 1:
            return powerlaw(
                data,
                alpha=kwds["alpha"],
                high=kwds["maximum"],
                low=kwds["minimum"],
            )
        else:
            mins = [kwds["minimum"]] + [
                kwds[f"break_{ii + 1}"] for ii in range(self.n_powerlaws - 1)
            ]
            maxs = [kwds[f"break_{ii + 1}"] for ii in range(self.n_powerlaws - 1)] + [
                kwds["maximum"]
            ]
            corrections = xp.cumprod(
                xp.array(
                    [1.0]
                    + [
                        powerlaw(
                            kwds[f"break_{ii}"],
                            alpha=kwds[f"alpha_{ii}"],
                            low=mins[ii - 1],
                            high=maxs[ii - 1],
                        )
                        / powerlaw(
                            kwds[f"break_{ii}"],
                            alpha=kwds[f"alpha_{ii + 1}"],
                            low=mins[ii],
                            high=maxs[ii],
                        )
                        for ii in range(1, self.n_powerlaws)
                    ]
                )
            )
            return xp.sum(
                xp.array(
                    [
                        corrections[ii]
                        * powerlaw(
                            data,
                            alpha=kwds[f"alpha_{ii + 1}"],
                            high=maxs[ii],
                            low=mins[ii],
                        )
                        for ii in range(self.n_powerlaws)
                    ]
                ),
                axis=0,
            ) / xp.sum(corrections)

    def __call__(self, data, **kwds):
        kwds = {
            self.replace_key(key.strip(self.name)): value
            for key, value in kwds.items()
            if key.startswith(self.name)
        }
        mix = xp.clip(
            xp.sum(
                xp.array([kwds[f"weight_{ii + 1}"] for ii in range(self.n_gaussians)])
            ),
            0,
            1,
        )
        return (1 - mix) * self.powerlaw_component(
            data, **kwds
        ) + mix * self.gaussian_component(data, **kwds)
