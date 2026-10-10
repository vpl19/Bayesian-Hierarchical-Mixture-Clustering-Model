"""Bayesian hierarchical mixture model for subgroup-specific phenotypes.

The model clusters individuals who are nested in ``J`` subgroups (for example
age groups) into ``K`` clusters that are comparable across subgroups but are
allowed to differ moderately between them.  For individual ``i`` in subgroup
``j``,

.. math::

    p(y_{ij}) = \\sum_{k=1}^{K} w_{kj}\\,
                \\mathcal{N}(y_{ij} \\mid \\mu_k + \\beta_k \\bar{y}_j, V_k),

where :math:`\\bar{y}_j` is the empirical mean of subgroup ``j`` and the data
have been standardised to global mean 0 and standard deviation 1.  The priors
are :math:`w_{\\cdot j} \\sim \\mathrm{Dirichlet}(1, \\ldots, 1)`,
:math:`\\mu_k \\sim \\mathcal{N}(0, 10 I)`,
:math:`\\beta_k \\sim \\mathcal{N}(0, 1)` and
:math:`R_k \\sim \\mathrm{LKJ}(1)`.

Two parameterisations of the cluster covariance are available, selected with
the ``estimate_diagonal`` flag of :func:`hierarchical_mixture_model`:

* ``estimate_diagonal=False`` (default, the model used in the paper) fixes the
  diagonal at :math:`V_k = R_k / K`.
* ``estimate_diagonal=True`` (the sensitivity analysis) estimates a single
  shared diagonal, :math:`V_k = \\sigma R_k` with
  :math:`\\sigma \\sim \\mathrm{HalfNormal}(1)`.

Latent cluster allocations are enumerated out during sampling, so the MCMC
samples only the continuous parameters.  Allocations are recovered afterwards
with :func:`posterior_allocations`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro import handlers
from numpyro.contrib.funsor import config_enumerate, infer_discrete
from numpyro.infer import MCMC, NUTS

__all__ = [
    "hierarchical_mixture_model",
    "subgroup_means",
    "run_mcmc",
    "posterior_allocations",
    "allocation_probabilities",
]


def subgroup_means(data: ArrayLike, subgroup: ArrayLike) -> jax.Array:
    """Compute the empirical mean of each subgroup.

    Args:
        data: Standardised observations, shape ``(n_observations, n_variables)``.
        subgroup: Integer subgroup label of each observation, shape
            ``(n_observations,)``.  Labels must be ``0, ..., n_subgroups - 1``.

    Returns:
        Subgroup means :math:`\\bar{y}_j`, shape ``(n_subgroups, n_variables)``.
    """
    data = jnp.asarray(data)
    subgroup = jnp.asarray(subgroup)
    n_subgroups = int(jnp.max(subgroup)) + 1
    return jnp.stack([data[subgroup == j].mean(axis=0) for j in range(n_subgroups)])


def hierarchical_mixture_model(
    data: ArrayLike,
    subgroup: ArrayLike,
    means: ArrayLike,
    n_clusters: int,
    estimate_diagonal: bool = False,
) -> None:
    """NumPyro model for the Bayesian hierarchical mixture model.

    Args:
        data: Standardised observations, shape ``(n_observations, n_variables)``.
        subgroup: Integer subgroup label of each observation, shape
            ``(n_observations,)``.
        means: Subgroup means from :func:`subgroup_means`, shape
            ``(n_subgroups, n_variables)``.
        n_clusters: Number of mixture components ``K``, fixed in advance.
        estimate_diagonal: If ``False`` (default) the cluster covariance is
            :math:`V_k = R_k / K`.  If ``True`` a single shared diagonal
            :math:`\\sigma \\sim \\mathrm{HalfNormal}(1)` is estimated and
            :math:`V_k = \\sigma R_k`.
    """
    data = jnp.asarray(data)
    means = jnp.asarray(means)
    n_variables = data.shape[-1]
    n_subgroups = means.shape[0]

    if estimate_diagonal:
        scale = numpyro.sample("scale", dist.HalfNormal(1.0))
    else:
        scale = 1.0 / n_clusters

    with numpyro.plate("clusters", n_clusters):
        beta = numpyro.sample("beta", dist.Normal(0.0, 1.0))
        mu = numpyro.sample(
            "mu",
            dist.MultivariateNormal(
                jnp.zeros(n_variables), 10.0 * jnp.eye(n_variables)
            ),
        )
        corr = numpyro.sample("corr", dist.LKJ(n_variables, concentration=1.0))

    with numpyro.plate("subgroups", n_subgroups):
        weights = numpyro.sample("weights", dist.Dirichlet(jnp.ones(n_clusters)))

    covariance = numpyro.deterministic("covariance", scale * corr)
    # Shared cluster mean plus a subgroup-specific perturbation,
    # mu_k + beta_k * ybar_j, shape (n_subgroups, n_clusters, n_variables).
    centres = numpyro.deterministic(
        "centres", mu + beta[..., :, None] * means[..., :, None, :]
    )

    with numpyro.plate("observations", data.shape[0]):
        allocation = numpyro.sample(
            "allocation",
            dist.Categorical(weights[subgroup]),
            infer={"enumerate": "parallel"},
        )
        numpyro.sample(
            "obs",
            dist.MultivariateNormal(
                centres[subgroup, allocation, :],
                covariance_matrix=covariance[allocation],
            ),
            obs=data,
        )


def run_mcmc(
    data: ArrayLike,
    subgroup: ArrayLike,
    means: ArrayLike,
    n_clusters: int,
    *,
    estimate_diagonal: bool = False,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    seed: int = 0,
    progress_bar: bool = True,
) -> MCMC:
    """Fit the model with the NUTS sampler.

    The posterior is highly multimodal, so in practice many chains are needed
    (see the README).  To run chains in parallel, call
    ``numpyro.set_host_device_count(num_chains)`` before this function;
    otherwise NumPyro falls back to running them sequentially.

    Args:
        data: Standardised observations, shape ``(n_observations, n_variables)``.
        subgroup: Integer subgroup label of each observation.
        means: Subgroup means from :func:`subgroup_means`.
        n_clusters: Number of mixture components ``K``.
        estimate_diagonal: Covariance parameterisation, see
            :func:`hierarchical_mixture_model`.
        num_warmup: Number of warmup iterations per chain.
        num_samples: Number of retained samples per chain.
        num_chains: Number of chains.
        seed: Seed for the pseudo-random number generator.
        progress_bar: Whether to display the sampler progress bar.

    Returns:
        The :class:`~numpyro.infer.MCMC` object after sampling.
    """
    mcmc = MCMC(
        NUTS(hierarchical_mixture_model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        data=data,
        subgroup=subgroup,
        means=means,
        n_clusters=n_clusters,
        estimate_diagonal=estimate_diagonal,
    )
    return mcmc


def posterior_allocations(
    mcmc: MCMC,
    data: ArrayLike,
    subgroup: ArrayLike,
    means: ArrayLike,
    n_clusters: int,
    *,
    estimate_diagonal: bool = False,
    seed: int = 0,
) -> jax.Array:
    """Draw cluster allocations from the posterior.

    The allocations are enumerated out during sampling, so they are recovered
    afterwards: for every posterior draw of the continuous parameters, the
    allocations are sampled from their conditional posterior given the data.

    Args:
        mcmc: A fitted :class:`~numpyro.infer.MCMC` object from
            :func:`run_mcmc`.
        data: The observations the model was fitted to.
        subgroup: The subgroup labels the model was fitted to.
        means: The subgroup means the model was fitted to.
        n_clusters: Number of mixture components ``K``.
        estimate_diagonal: Covariance parameterisation, see
            :func:`hierarchical_mixture_model`.
        seed: Seed for the pseudo-random number generator.

    Returns:
        Cluster allocations, shape ``(n_draws, n_observations)``.  Cluster
        labels are only meaningful within a draw: the mixture is invariant to
        relabelling, so labels may switch between draws and between chains.
    """
    samples = mcmc.get_samples()

    def single_draw(rng_key: jax.Array, draw: dict[str, jax.Array]) -> jax.Array:
        def substitute_fn(site: dict) -> jax.Array | None:
            # Deterministic sites are recomputed rather than substituted.
            if site["type"] == "deterministic":
                return None
            return draw.get(site["name"])

        conditioned = config_enumerate(
            handlers.substitute(hierarchical_mixture_model, substitute_fn=substitute_fn)
        )
        model = infer_discrete(conditioned, rng_key=rng_key, temperature=1)
        model_trace = handlers.trace(handlers.seed(model, rng_key)).get_trace(
            data=data,
            subgroup=subgroup,
            means=means,
            n_clusters=n_clusters,
            estimate_diagonal=estimate_diagonal,
        )
        return model_trace["allocation"]["value"]

    n_draws = jax.tree.leaves(samples)[0].shape[0]
    rng_keys = jax.random.split(jax.random.PRNGKey(seed), n_draws)
    return jax.vmap(single_draw)(rng_keys, samples)


def allocation_probabilities(allocations: ArrayLike, n_clusters: int) -> jax.Array:
    """Posterior probability that each observation belongs to each cluster.

    Cluster labels are only identified up to relabelling, so this is only
    meaningful for draws whose labels are aligned, in practice the draws of a
    single chain.  Combining many chains requires an explicit relabelling step
    such as the consensus clustering described in the README.

    Args:
        allocations: Cluster allocations from :func:`posterior_allocations`,
            shape ``(n_draws, n_observations)``.
        n_clusters: Number of mixture components ``K``.

    Returns:
        Allocation probabilities, shape ``(n_observations, n_clusters)``, with
        rows summing to 1.
    """
    allocations = jnp.asarray(allocations)
    return jnp.stack(
        [(allocations == k).mean(axis=0) for k in range(n_clusters)], axis=-1
    )
