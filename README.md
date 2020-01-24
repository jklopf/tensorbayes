# TensorBayes: TensorFlow application of Bayesian penalized regression
Repository for Master Thesis of Science in Molecular Life Sciences, specialisation in Bioinformatics, UNIL, by Jonathan Klopfenstein.

## Description
TensorBayes is a TensorFlow implementation of a Gibbs sampling algorithm for Bayesian penalized regression with a spike and slab prior, intended for genome wide assotiation studies. The Bayesian mixed linear model is as follows: with N individuals and M markers,  
![equation](http://bit.ly/37qsdYe)
where y is the vector of phenotype, x is the genotype matrix, beta is the vector of the marker effect sizes and epsilon the vector of the residual effects.

## Abstract
In the context of quantitative genetics, Bayesian learning models have been
shown to out-compete frequentist approaches by estimating the effects of
genetic markers jointly and conditionally on all markers, alleviating model
overtting by correcting for correlation between markers and enabling the
discovery of new risk factors of smaller effects. However, with increasing
collections of genotype and phenotype data, challenge now resides in
the development of efficient computing methods for ever-growing large-scale
datasets, impeding research in both computation time and infrastructure
costs. With this paradigm, a recent framework, GPU-accelerated computing,
has shown increased computing efficiency through large-scale parallelization.
For this project, a MCMC Gibbs sampling algorithm for Bayesian penalized
regression has been implemented and developed using TensorFlow, Google's
machine-learning system for CPUs and GPUs. The application is benchmarked
for both types of hardware acceleration and the usage of TensorFlow
for Gibbs sampling algorithms is discussed.

## Full thesis
The full thesis is available for download at this [link](https://drive.google.com/file/d/1ojLLjuzZpHP5mlUQpXt7UFQBjvRI3v07/view?usp=sharing).

## Archive
[Previous README (outdated)]()

