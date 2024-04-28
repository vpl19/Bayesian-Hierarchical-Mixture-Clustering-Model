# Third-party imports
import jax.numpy as jnp
import numpy as np
import pandas as pd
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from jax import random

# Enable x64 mode for JAX to improve performance and set the number of cores
numpyro.enable_x64()
numpyro.set_host_device_count(5)


def load_data():
    """Load the dataset and prepare it for the model."""
    men = pd.read_csv(
        ".../data_pyro_men.csv",
        index_col=0,
    )
    men_data = men[
        [
            "height",
            "bmi",
            "WHtR",
            "hba1c",
            "hdl",
            "non_hdl",
            "sbp",
            "dbp",
            "eGFR",
            "pulse",
        ]
    ].to_numpy()
    men_label = men[["age_group"]].to_numpy().flatten()
    return men_data, men_label


def compute_group_means(data):
    """Compute the mean values for each age group in the dataset."""
    mean_values = data.groupby("age_group").mean().reset_index()
    means = jnp.array(
        [
            mean_values.loc[
                mean_values["age_group"] == i,
                [
                    "height",
                    "bmi",
                    "WHtR",
                    "hba1c",
                    "hdl",
                    "non_hdl",
                    "sbp",
                    "dbp",
                    "eGFR",
                    "pulse",
                ],
            ].to_numpy()
            for i in range(3)
        ]
    )
    return means.squeeze()


def model_HGMM(K, dimension, data, label, means):
    """Define the hierarchical model."""
    # Expanded means to match the dimensionality required for the model
    means_expanded = jnp.expand_dims(means, axis=1)
    means_repeated = jnp.repeat(means_expanded, K, axis=1)

    length = len(np.unique(label))

    with numpyro.plate("components", K):
        beta = numpyro.sample("beta", dist.Normal(1, 1))
        locs = numpyro.sample(
            "locs",
            dist.MultivariateNormal(jnp.zeros(dimension), 10 * jnp.eye(dimension)),
        )
        corr_mat = numpyro.sample("corr_mat", dist.LKJ(dimension, concentration=1))

    with numpyro.plate("Age group", length):
        cluster_proba = numpyro.sample("cluster_proba", dist.Dirichlet(jnp.ones(K)))

    sigma = numpyro.deterministic("sigma", 1 / K * corr_mat)
    beta_expanded = jnp.expand_dims(beta, axis=1)
    beta_repeated = jnp.repeat(beta_expanded, dimension, axis=1)
    locs_perturb_adjusted = numpyro.deterministic(
        "locs_perturb_adjusted", locs + beta_repeated * means_repeated
    )

    with numpyro.plate("data", len(data)):
        assignment = numpyro.sample(
            "assignment",
            dist.Categorical(cluster_proba[label]),
            infer={"enumerate": "parallel"},
        )
        numpyro.sample(
            "obs",
            dist.MultivariateNormal(
                locs_perturb_adjusted[label, assignment, :],
                covariance_matrix=sigma[assignment],
            ),
            obs=data,
        )


def model_HGMM_diag_est(K, dimension=10, data=None, label=None):
    length = len(np.unique(label))
    means_ex2 = jnp.repeat(
        means_ex[:, :], K, axis=1
    )  # same prior for each cluster within age groups, thus the repeat
    variance = numpyro.sample("variance", dist.HalfNormal(scale=1 / K))
    with numpyro.plate("components", K):
        beta = numpyro.sample("beta", dist.Normal(0, 1))
        locs = numpyro.sample(
            "locs",
            dist.MultivariateNormal(jnp.zeros(dimension), 10 * jnp.eye(dimension)),
        )
        corr_mat = numpyro.sample("corr_mat", dist.LKJ(dimension, concentration=1))
    with numpyro.plate("Age group", length):
        cluster_proba = numpyro.sample("cluster_proba", dist.Dirichlet(jnp.ones(K)))
    sigma = numpyro.deterministic("sigma", variance * corr_mat)
    beta_ex = jnp.expand_dims(beta, axis=1)
    beta_ex2 = jnp.repeat(beta_ex, dimension, axis=1)
    locs_perturb_adjusted = numpyro.deterministic(
        "locs_perturb_adjusted",
        locs
        + beta_ex2
        * means_ex2,  ################################### Consider saving the beta_ex2*means_ex2 as locs perturb_adjusted to be able to post process like the rest
    )  # regression on component center based on subgroup empirical means
    with numpyro.plate("data", len(data)):
        assignment = numpyro.sample(
            "assignment",
            dist.Categorical(cluster_proba[label]),
            infer={"enumerate": "parallel"},
        )
        numpyro.sample(
            "obs",
            dist.MultivariateNormal(
                locs_perturb_adjusted[label, assignment, :],
                covariance_matrix=sigma[assignment],
            ),
            obs=data,
        )


def run_mcmc(
    rng_key, model, men_data, men_label, means, n_clusters, n_variables, **mcmc_kwargs
):
    """Run the MCMC algorithm for the defined model and perform posterior predictive checks."""
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        **mcmc_kwargs,
    )

    mcmc.run(
        rng_key,
        K=n_clusters,
        dimension=n_variables,
        data=men_data,
        label=men_label,
        means=means,
    )
    posterior_samples = mcmc.get_samples(group_by_chain=True)

    # Perform posterior predictive checks
    posterior_predictive = Predictive(
        model, posterior_samples, infer_discrete=True, batch_ndims=2
    )
    posterior_predictions = posterior_predictive(
        rng_key, K=n_clusters, dimension=n_variables, data=men_data, label=men_label
    )

    return posterior_samples, posterior_predictions


def save_results(posterior_samples, posterior_predictions):
    """Save the results from the MCMC algorithm and posterior predictions."""
    np.save("locs_men.npy", posterior_samples["locs"])
    np.save("rho_men.npy", posterior_samples["sigma"])
    np.save("cluster_proba_men.npy", posterior_samples["cluster_proba"])
    np.save("beta.npy", posterior_samples["beta"])
    np.save("locs_perturb.npy", posterior_samples["locs_perturb_adjusted"])
    np.save("cluster_membership.npy", posterior_predictions["assignment"])


if __name__ == "__main__":
    # Load and prepare data
    men_data, men_label = load_data()
    means = compute_group_means(
        pd.read_csv(
            ".../data_pyro_men.csv",
            index_col=0,
        )
    )

    # Run the model and perform posterior predictive checks
    posterior_samples, posterior_predictions = run_mcmc(
        rng_key=random.PRNGKey(1234),
        # can be changed to `model_HGMM_diag_est` if wish to estimate the diagonal of the covariance matrix
        model=model_HGMM,
        men_data=men_data,
        men_label=men_label,
        means=means,
        n_variables=10,
        n_clusters=10,
        # mcmc parameters
        num_warmup=1000,
        num_samples=2000,
        num_chains=50,
        chain_method="parallel",
    )

    # Save results
    save_results(posterior_samples, posterior_predictions)

    print("Model run and posterior predictive checks complete.")
