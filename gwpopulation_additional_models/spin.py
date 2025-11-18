from gwpopulation.utils import truncnorm


def truncated_normal_spin_magnitude_independent(
    dataset,
    mu_chi_1,
    mu_chi_2,
    sigma_chi_1,
    sigma_chi_2,
):
    return truncnorm(
        dataset["a_1"], mu=mu_chi_1, sigma=sigma_chi_1, low=0, high=1
    ) * truncnorm(dataset["a_2"], mu=mu_chi_2, sigma=sigma_chi_2, low=0, high=1)


def truncated_normal_spin_magnitude_iid(dataset, mu_chi, sigma_chi):
    return truncated_normal_spin_magnitude_independent(
        dataset,
        mu_chi_1=mu_chi,
        mu_chi_2=mu_chi,
        sigma_chi_1=sigma_chi,
        sigma_chi_2=sigma_chi,
    )
