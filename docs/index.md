# Overview


## Overview
Arborist is a method to rank SNV clone trees inferred from bulk DNA sequencing data by leveraging low-pass single-cell DNA sequencing (scDNA-seq) data. The method is designed to prioritize the most probable tree, helping to resolve ambiguities in bulk tree inference. Arborist uses variational inference to compute the evidence lower bound on the posterior probability of each tree in the input candidate set and approximates the cell to clone assignment posterior distribution as well as the SNV to SNV cluster assignment posterior. Arborist not only helps to resolve tree ambiguity within the bulk solution space but also helps improve an SNV cluster and yields a natural way to genotype single-cells and derive cell-to-clone assignments for downstream analysis.

For the code repository, visit [Github](https://github.com/VanLoo-lab/Arborist).

## Description

* `arborist -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
