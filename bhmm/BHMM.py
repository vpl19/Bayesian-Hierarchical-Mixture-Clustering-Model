# Third-party imports
import jax
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


def load_data(filepath):
    return pd.read_csv(filepath, index_col=0)


def compute_group_means(data, age_groups, columns):
    means = []
    for group in age_groups:
        mean_values = data[data["age_group"] == group][columns].mean().to_numpy()
        means.append(mean_values)
    means_array = jnp.asarray(means)
    means_expanded = jnp.expand_dims(means_array, axis=1)
    return means_expanded


def prepare_data_labels(data, feature_columns, label_column):
    data_features = data[feature_columns].to_numpy()
    labels = data[[label_column]].to_numpy().flatten()
    return data_features, labels


file_path = "data_pyro_men_subsample.csv"
feature_columns = [
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
label_column = "age_group"
age_groups = [0, 1, 2]  # Assuming age groups are 0, 1, 2

# Load data
men = load_data(file_path)

# Prepare data and labels
men_data, men_label = prepare_data_labels(men, feature_columns, label_column)

# Compute group means
group_means = compute_group_means(men, age_groups, feature_columns)


def model_HGMM(K, dimension, data, label, means):
    means_repeated = jnp.repeat(means[:, :], K, axis=1)
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


def model_HGMM_diag_est(K, dimension, data, label, means):
    means_repeated = jnp.repeat(means[:, :], K, axis=1)
    length = len(np.unique(label))

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
        locs + beta_ex2 * means_repeated,
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


kernel = NUTS(model_HGMM)
num_warmup, num_samples = 1000, 1000
n_variables = 10
n_clusters = 10
num_chains = 2  # increase if possible

mcmc = MCMC(
    kernel,
    num_warmup=num_warmup,
    num_samples=num_samples,
    num_chains=num_chains,  # mchain_method='parallel',
)

rng_key = jax.random.PRNGKey(1234)

mcmc.run(
    rng_key,
    K=n_clusters,
    dimension=10,
    data=men_data,
    label=men_label,
    means=group_means,
)

posterior_samples = mcmc.get_samples(group_by_chain=True)


np.save("locs_men.npy", posterior_samples["locs"])
np.save("rho_men.npy", posterior_samples["sigma"])
np.save("cluster_proba_men.npy", posterior_samples["cluster_proba"])
np.save("beta.npy", posterior_samples["beta"])
np.save("locs_perturb.npy", posterior_samples["locs_perturb_adjusted"])

print(" ")
print(" ")
print("DONE")
print(" ")
print(" ")
posterior_predictive = Predictive(
    model_HGMM, posterior_samples, infer_discrete=True, batch_ndims=2
)

posterior_predictions = posterior_predictive(
    rng_key,
    K=n_clusters,
    dimension=n_variables,
    data=men_data,
    label=men_label,
)
np.save("cluster_membership.npy", posterior_predictions["assignment"])

print("Model run and posterior predictive checks complete.")
