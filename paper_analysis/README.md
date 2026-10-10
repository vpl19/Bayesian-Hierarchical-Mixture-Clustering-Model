# The analysis in the paper

`run_nhanes_chains.py` fits the model to men aged 20 and over in NHANES, clustering
them into K = 10 anthropometric, cardiometabolic and renal phenotypes within three age groups, using the 10
risk factors listed below. It draws 50 chains in total (10 runs of 5 parallel
chains, each 1000 warmup and 1000 sampling iterations), because the posterior of a
mixture of this size is highly multimodal and a single chain explores only part of
it. Each run writes a folder containing the posterior samples grouped by chain and
the cluster allocations of every draw.

This takes several days on a high-performance cluster. It is not something to run on
a laptop; for that, see the bivariate example in
[`examples/simulated_example.ipynb`](../examples/simulated_example.ipynb).

Set `DATA_PATH` and `OUTPUT_DIR` at the top of the script before running it.

## Data

The data are not in this repository. The cleaned NHANES data and the cleaning code
are in the Zenodo record of our earlier paper,
<https://zenodo.org/records/10075388> (Lhoste et al., 2023, *Nature Cardiovascular
Research*, <https://doi.org/10.1038/s44161-023-00391-y>). That record contains:

- `NHANES_Cleaned_single.RDS`, the merged NHANES surveys from 1988 to 2018 with
  single-variable cleaning applied;
- `1_Mahalanobis_cleaning.R`, the multivariate cleaning step.

To build the input file this script expects, starting from those two:

1. apply the Mahalanobis cleaning;
2. keep men (`sex == 1`) aged 20 and over;
3. compute non-HDL cholesterol (total cholesterol minus HDL) and the
   waist-to-height ratio;
4. keep individuals with all 10 risk factors measured;
5. create `age_group` as 0 for ages 20 to 39, 1 for 40 to 59 and 2 for 60 and over;
6. standardise each of the 10 risk factors to a mean of 0 and a standard deviation
   of 1.

The result is a CSV with a leading index column (the script reads it with
`index_col=0`) and the columns:

```
height, bmi, WHtR, hba1c, hdl, non_hdl, sbp, dbp, eGFR, pulse, age_group
```

The multivariate cleaning has a random element, so a file rebuilt this way may
differ from the one used in the paper by a handful of individuals.

## Not included

- Combining the 50 chains into a single partition by consensus clustering, done with
  `mcclust::maxpear` in R.
- The convergence diagnostics and the figures.

See the paper for these.
