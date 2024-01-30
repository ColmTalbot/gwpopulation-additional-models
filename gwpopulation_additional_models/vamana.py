from gwpopulation.utils import powerlaw, truncnorm, xp


class Vamana:
    """
    A basic implementation of the model described in https://arxiv.org/abs/2006.15047
    """

    def __init__(self, n_components, base_model=None, reference_parameters=None):
        self.n_components = n_components
        self.base_model = base_model
        self.reference_parameters = reference_parameters
        self.chirp_masses = xp.linspace(2, 100, 1000)
        self.mmin = self.chirp_masses[0]
        self.mmax = self.chirp_masses[-1]

    @property
    def variable_names(self):
        names = [f"weight_{ii}" for ii in range(self.n_components - 1)]
        names += [f"mu_m_{ii}" for ii in range(self.n_components - 1)]
        names += [f"sigma_m_{ii}" for ii in range(self.n_components)]
        names += [f"mu_sz_{ii}" for ii in range(self.n_components)]
        names += [f"sigma_sz_{ii}" for ii in range(self.n_components)]
        names += [f"alpha_q_{ii}" for ii in range(self.n_components)]
        names += [f"qmin_{ii}" for ii in range(self.n_components)]
        return names

    def reference_model(self, mass):
        if self.base_model is None:
            return xp.ones(mass.shape)
        elif self.reference_parameters is None:
            raise ValueError("No reference parameters provided for base_model")
        return self.base_model(mass, **self.reference_parameters)

    def __call__(self, dataset, **kwargs):
        import numpy as np

        prob = xp.zeros(dataset["chirp_mass"].shape)
        final_weight = 1 - sum(
            [kwargs[f"weight_{ii}"] for ii in range(self.n_components - 1)]
        )
        if final_weight < 0:
            return prob
        else:
            kwargs[f"weight_{self.n_components - 1}"] = final_weight
        self.base_prob = self.reference_model(dataset["chirp_mass"])
        self.norm_base_prob = self.reference_model(self.chirp_masses)
        masses = [kwargs[f"mu_m_{ii}"] for ii in range(self.n_components - 1)]
        masses.append(1 - sum(masses))
        masses = np.cumsum(masses)
        for ii in range(self.n_components):
            kwargs[f"mu_m_{ii}"] = self.mmin * (self.mmax / self.mmin) ** masses[ii]
            prob += (
                kwargs[f"weight_{ii}"]
                * self.p_mc(dataset["chirp_mass"], kwargs, ii)
                * self.p_chi(dataset["chi_1"], kwargs, ii)
                * self.p_chi(dataset["chi_2"], kwargs, ii)
                * self.p_mass_ratio(dataset["mass_ratio"], kwargs, ii)
            )
        return prob

    def p_mc(self, mass, kwargs, ii):
        prob = truncnorm(
            mass,
            mu=kwargs[f"mu_m_{ii}"],
            sigma=kwargs[f"sigma_m_{ii}"] * kwargs[f"mu_m_{ii}"],
            high=100,
            low=2,
        )
        if self.base_model is not None:
            norm_prob = xp.trapz(
                self.norm_base_prob
                * truncnorm(
                    self.chirp_mass,
                    mu=kwargs[f"mu_m_{ii}"],
                    sigma=kwargs[f"sigma_m_{ii}"] * kwargs[f"mu_m_{ii}"],
                    high=100,
                    low=2,
                ),
                self.chirp_mass,
            )
            prob *= self.base_prob / norm_prob
        return prob

    def p_chi(self, spin, kwargs, ii):
        return truncnorm(
            spin,
            mu=kwargs[f"mu_sz_{ii}"],
            sigma=kwargs[f"sigma_sz_{ii}"],
            high=1,
            low=-1,
        )

    def p_mass_ratio(self, mass_ratio, kwargs, ii):
        return powerlaw(
            mass_ratio,
            alpha=kwargs[f"alpha_q_{ii}"],
            high=1,
            low=kwargs[f"qmin_{ii}"],
        )
