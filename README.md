# Bayesian Hierarchical Mixture Clustering Model

Clustering model designed to find cardiometabolic phenotypes in different subgroups of a population.


## Reply TO DO

- *50 chains???*
    - I changed it to 2 chains for now, in the paper I used 50 chains because it was estimating a highly multimodal posterior
- *Where do I get ".../data_pyro_men.csv"*
    -I have added a subsample of the data, i can ask Majid if he is okay for us to add it here (the data is public but we have done some pre-processing to it, it is a big pain to publish the preprocessing scripty because it is done within NCD-risc framework and has loads that should be removed before publishing, the cleaning also has a stochastic element that can change the results by one or two individuals depending on the R version... But more importantly there would be no way for someone to run the code on a laptop on the full data, even for just one chain it would take ages. On top of that if we want to do it like the paper then there is the consensus clustering that is done in R, i should probably add the trace plots, (that I have done in R) as well to checl it converge etc. I have actually done a repo already with everything: https://github.com/vpl19/BHMM_HPC, but it is very ugly and I just did it so that Sarkaaj a new PhD student from Majid can reproduce what I did if needed as he will use the model)
- *test the script because you have the data*
    - Done, works but I have changed a few things on how the means are calculated and remove the if main=main and the run_mcmc funtion as I thought it was making things more complex than it should.
- *test the notebook because you have the notebook*
    - Done
- *where is n_clusters defined?*
    -Added, it is pre-specified in the model

- *where is means_ex defined?*
    -that was a mistake on my end. Changed it

- *replace the stuff in the notebook with `from bhmm.BHMM import model_HGMM, run_mcmc` etc where possible*
    -Because I removed the if main=main etc I haven't. I quite like the idea that both BHMM.py and the notebook work as stand alone. 
    Alternative would be to have a separate script with the two models and import it in both BHMM.py and the notebook, or change again to mnain etc. 
    Would be worth discussing quickly. 


What is the liscence thing that you added, is that a standard thing?

## Model Aims

In the process of clustering multiple similar subgroups, two main strategies can be employed: clustering jointly and clustering separately. 
Joint clustering involves the simultaneous analysis of all subgroups to identify common patterns or groups, effectively treating
the combined data as a singular entity. In contrast, separate clustering entails analyzing each subgroup independently, without considering potential correlations or commonalities
between them. While clustering jointly will aim at partitioning all individuals aggregated together rather than
finding partitions within each group, clustering each subgroup separately captures the unique characteristics of each subgroup but comparison of the results is at best impractical. 


The Bayesian Hierarchical Mixture Clustering Model I have developed here seeks an in-between of clustering separately and jointly. 
This model can identify comparable phenotypes across subgroups while also capturing subgroup specificities. 
Specifically the aim of this model are: 

- To identify comparable phenotypes across multiple subgroups. To facilitate meaningful comparisons the model must identify analogous phenotypes across subgroups
that can be matched one by one.
- To capture subgroup specificities. Specifically, the characteristics of the identified
phenotypes and their prevalence should be able to differ moderately between subgroups if suggested by the data while remaining comparable.
- Because similar relationships between risk factors are expected across subgroups due to shared biological processes, the model should be able to borrow some of the
shared information between subgroups.

## Model Features and assumptions

- There are similar cardiometabolic and renal phenotypes across subgroups with different prevalences, therefore the weights of the mixture will be both cluster and
subgroup-specific.
- The same cardiometabolic and renal phenotype may have slightly different mean risk factor levels across subgroups, therefore cluster means are allowed to differ
across subgroups by introducing a small perturbation that is added to shared global cluster means.
- The more difference there is between a subgroup and the rest of the population the more difference we can expect on its phenotypes, therefore the perturbation 
should be dependent on the difference between subgroup means yj and overall mean.
- Similar correlations between risk factors are expected across subgroups but not necessarily across clusters therefore the covariance matrix is cluster-specific but the
shared across subgroups

## Model Specification

![Model Equation](Model_specifications.png)

## Usage

All variables are scaled to have a global mean of 0 and a standard deviation of 1 before clustering.
The code provided in BHMM.py contains the model and an example on how it should be run on a data set "data_pyro_men.csv" containing 10 cardiometabolic phenotypes and separated in 3 age groups.
This dataset was extracted from the publically available NHANES surveys and preprocessed as done in the following study: https://www.nature.com/articles/s44161-023-00391-y

An example of the application of the model to simulated data is provided in the notebook: BHMM_simulated_bivariate_data.ipynb
Run `pip install .` to install the required packages.
