# CLUSTERING OF MONOAMINERGIC NEURONS

This is the repository for a project focusing on unsupervised clustering of gene expression data. To be more specific, we work with monoaminegic neurons in drosophila (fruit flies), and are attempting two main tasks:

1.  splitting the data set into 4 classes, based on which neurotransmitter the neurons are specialized for (dopamine, serotonin, octopamine or tyramine),
2.  trying to split the class of dopaminergic neurons further into smaller subclasses (in an exploratory nature, we do not know what could these classes mean).

Tha data we use was part of a larger study of the impact of water deprivation on brain cells in drosophila ([link to paper](https://www.cell.com/current-biology/pdf/S0960-9822(22)01175-7.pdf), [link to Github repository](https://github.com/sims-lab/FlyThirst)). As it was shared with us through private channels, we unfortunately cannot share it. Similar is true for the code for `kcluster` clustering (Bobrowski & Škraba, ["Cluster-Persistence for Weighted Graphs"](https://arxiv.org/pdf/2310.00350.pdf)) which we use extensively.

The repository is organized as follows: 
 * Custom methods written for this specific data set are collected in `clustering_methods.py`.
 * Most code is written in Jupyter notebooks:
   -  `monoaminergic_clustering.ipynb` contains results of classifying monoaminergic neurons into four subtypes.
   -  `dopaminergic_clustering.ipynb` contains results of of clustering dopaminergic neurons.
 * Folder `Images` contains images used in above notebooks.
 * Folder `Pickle` contains pickled files with commonly used variables.
