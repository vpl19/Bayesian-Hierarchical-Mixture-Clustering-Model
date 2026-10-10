"""Bayesian hierarchical mixture model for subgroup-specific phenotypes."""

from bhmm.model import (
    allocation_probabilities,
    hierarchical_mixture_model,
    posterior_allocations,
    run_mcmc,
    subgroup_means,
)

__version__ = "0.1.0"

__all__ = [
    "hierarchical_mixture_model",
    "subgroup_means",
    "run_mcmc",
    "posterior_allocations",
    "allocation_probabilities",
]
