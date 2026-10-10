"""Fit the hierarchical mixture model to the NHANES data used in the paper.

Clusters men aged 20 and over from NHANES into K = 10 anthropometric,
cardiometabolic and renal phenotypes, within three age groups, using the 10 risk
factors listed in RISK_FACTORS. The posterior is highly multimodal, so the analysis
in the paper draws 50 chains in total: 10 runs of 5 parallel chains, each chain
1000 warmup and 1000 sampling iterations. One output folder is written per run,
holding the posterior samples grouped by chain and the cluster allocations of every
draw.

This takes several days on a high-performance cluster. The 50 chains are then
combined by consensus clustering, which is not part of this script; see
paper_analysis/README.md.

Set DATA_PATH and OUTPUT_DIR below before running.
"""

from pathlib import Path

import numpy as np
import numpyro
import pandas as pd

from bhmm import posterior_allocations, run_mcmc, subgroup_means

# Input file: one row per individual, the 10 risk factors standardised to a global
# mean of 0 and a standard deviation of 1, plus an age_group column (0, 1, 2).
# See paper_analysis/README.md for how to build it.
DATA_PATH = Path("path/to/data_pyro_men.csv")
OUTPUT_DIR = Path("path/to/results")

RISK_FACTORS = [
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
SUBGROUP_COLUMN = "age_group"

N_CLUSTERS = 10
N_RUNS = 10
CHAINS_PER_RUN = 5
NUM_WARMUP = 1000
NUM_SAMPLES = 1000

# The 5 chains of a run are drawn in parallel, one per host device.
numpyro.set_host_device_count(CHAINS_PER_RUN)
numpyro.enable_x64()


def main() -> None:
    men = pd.read_csv(DATA_PATH, index_col=0)
    data = men[RISK_FACTORS].to_numpy()
    subgroup = men[SUBGROUP_COLUMN].to_numpy().flatten()
    means = subgroup_means(data, subgroup)

    for run in range(1, N_RUNS + 1):
        print(f"run {run} of {N_RUNS}", flush=True)
        run_dir = OUTPUT_DIR / f"Run{run}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # The model and its priors are defined in bhmm/model.py (function
        # hierarchical_mixture_model), as specified in the paper:
        # beta_k ~ N(0, 1), mu_k ~ N(0, 10 I), R_k ~ LKJ(1), V_k = R_k / K and
        # w_.j ~ Dirichlet(1, ..., 1). run_mcmc fits it with the NUTS sampler.
        # A different seed per run, so the runs start from different points of
        # the posterior. Initialisation is left at the NumPyro default.
        mcmc = run_mcmc(
            data,
            subgroup,
            means,
            N_CLUSTERS,
            num_warmup=NUM_WARMUP,
            num_samples=NUM_SAMPLES,
            num_chains=CHAINS_PER_RUN,
            seed=run + 1,
        )

        samples = mcmc.get_samples(group_by_chain=True)
        for site in ("mu", "beta", "weights", "covariance", "centres"):
            np.save(run_dir / f"{site}.npy", np.asarray(samples[site]))

        # Cluster allocations for every draw, reshaped to (chain, draw, individual)
        # to match the samples saved above.
        allocations = np.asarray(
            posterior_allocations(mcmc, data, subgroup, means, N_CLUSTERS, seed=run + 1)
        )
        np.save(
            run_dir / "allocations.npy",
            allocations.reshape(CHAINS_PER_RUN, NUM_SAMPLES, -1),
        )

    print("done", flush=True)


if __name__ == "__main__":
    main()
