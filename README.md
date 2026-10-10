# Bayesian hierarchical mixture model for subgroup-specific phenotypes

Implementation in [NumPyro](https://num.pyro.ai) of the clustering model from
[Lhoste et al. (2026)](https://doi.org/10.1093/jrsssa/qnag109), with a worked example
on simulated data. The model was developed to find anthropometric, cardiometabolic and
renal phenotypes in different subgroups of a population. The script used for the
analysis in the paper is in [`paper_analysis/`](paper_analysis/).

## Aim

When several similar subgroups of a population have to be clustered, the two obvious
strategies both lose something. Clustering all subgroups jointly partitions the
pooled individuals rather than finding a partition within each subgroup, so it
cannot express how a phenotype differs from one subgroup to the next. Clustering
each subgroup separately captures those specificities but produces partitions that
are, at best, impractical to compare. This model sits in between the two: it finds
phenotypes that can be matched one to one across subgroups, so they remain
comparable, while letting their prevalence and their mean risk factor levels differ
moderately between subgroups where the data suggest it, and borrowing the
information that subgroups share because of common biological processes.

## The model

Let $y_{ij}$ be the vector of risk factors of individual $i$ in subgroup $j$, with
all variables standardised to a global mean of 0 and a standard deviation of 1, and
let $\bar{y}_j$ be the empirical mean of subgroup $j$. For $K$ clusters,

$$p(y_{ij}) = \sum_{k=1}^{K} w_{kj} \, \mathrm{N}\left(y_{ij} \mid \mu_k + \beta_k \bar{y}_j,\; V_k\right).$$

The mixture weights $w_{kj}$ are both cluster- and subgroup-specific, so the same
phenotype can be more or less prevalent in different subgroups. The cluster mean is
a shared location $\mu_k$ plus a perturbation $\beta_k \bar{y}_j$, so a phenotype
shifts with how far its subgroup sits from the population as a whole. The covariance
$V_k$ is cluster-specific but shared across subgroups, reflecting the expectation
that relationships between risk factors differ between phenotypes but not between
subgroups.

The priors are

$$w_{\cdot j} \sim \mathrm{Dirichlet}(1, \ldots, 1), \qquad
\mu_k \sim \mathrm{N}(0, 10 I), \qquad
\beta_k \sim \mathrm{N}(0, 1), \qquad
R_k \sim \mathrm{LKJ}(1),$$

where $R_k$ is a correlation matrix. In the main model the diagonal is fixed,

$$V_k = R_k / K,$$

which keeps clusters of comparable size; in the paper's sensitivity analysis,
estimating the diagonal led to one broad cluster absorbing most individuals and the
others being nearly empty. That sensitivity analysis estimates a single shared
diagonal,

$$V_k = \sigma R_k, \qquad \sigma \sim \mathrm{HalfNormal}(1),$$

available through the `estimate_diagonal=True` flag.

The number of clusters $K$ is fixed in advance. The discrete cluster allocations are
enumerated out, so the sampler only moves over the continuous parameters;
allocations are recovered afterwards.

## What's in this repository

- [`bhmm/model.py`](bhmm/model.py): the model (`hierarchical_mixture_model`) and
  helper functions to compute the subgroup means (`subgroup_means`), fit the model
  with the NUTS sampler (`run_mcmc`) and recover the cluster allocations
  (`posterior_allocations`, `allocation_probabilities`).
- [`examples/simulated_example.ipynb`](examples/simulated_example.ipynb): a worked
  example on simulated bivariate data with three subgroups and two clusters. It
  simulates the data, fits the model and compares the estimated clusters with the
  truth, and runs in a couple of minutes on a laptop. This is the best place to start.
- [`paper_analysis/`](paper_analysis/): the script used for the analysis in the
  paper, with notes on how to obtain the data (see below).

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/vpl19/Bayesian-Hierarchical-Mixture-Clustering-Model.git
cd Bayesian-Hierarchical-Mixture-Clustering-Model
pip install -e ".[notebook]"
```

The `notebook` extra adds Jupyter, which is only needed to run the example. For the
model alone, `pip install -e .` is enough.

## The analysis in the paper

The paper clusters men aged 20 and over in NHANES into 10 anthropometric,
cardiometabolic and renal phenotypes across three age groups, using 10 risk factors.
Because the posterior of a mixture that size is highly multimodal, it draws 50 chains
and combines them into a single partition by consensus clustering, which takes
several days on a high-performance cluster.

The script that produced those chains, and notes on how to obtain the data, are in
[`paper_analysis/`](paper_analysis/).

## How to cite

```bibtex
@article{10.1093/jrsssa/qnag109,
    author = {Lhoste, Victor P F and Fan, Yefeng and Bennett, James E and Filippi, Sarah and Paciorek, Christopher J and Wang, Junyang and Zhou, Bin and Rashid, Theo and Phelps, Nowell H and Ezzati, Majid},
    title = {Hierarchical Bayesian mixture model for subgroup-specific cardiometabolic phenotypes},
    journal = {Journal of the Royal Statistical Society Series A: Statistics in Society},
    pages = {qnag109},
    year = {2026},
    month = {10},
    issn = {0964-1998},
    doi = {10.1093/jrsssa/qnag109},
    url = {https://doi.org/10.1093/jrsssa/qnag109},
}
```

## Use of AI tools

This repository was tidied up in October 2026 with the help of Claude Code
(Anthropic), an AI coding assistant, which was used to restructure the code and
documentation. The model and methods are unchanged from those described in the paper,
and all changes were reviewed by Victor Lhoste.

## Acknowledgements

Thanks to [Theo Rashid](https://github.com/theorashid) for his help implementing the
model in NumPyro.
