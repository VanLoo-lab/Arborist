# Overview
Arborist is a method to rank SNV phylogenetic trees inferred from bulk DNA sequencing data by leveraging low-pass single-cell DNA sequencing (scDNA-seq) data. The method is designed to prioritize the most probable tree, helping to resolve ambiguities in bulk tree inference. Arborist uses variational inference to compute the evidence lower bound on the posterior probability of each tree in the input candidate set and approximates the cell to clone assignment posterior distribution as well as the SNV to SNV cluster assignment posterior. Thus, Arborist not only helps to resolve tree ambiguity within the bulk solution space but also helps improve an SNV cluster and yields a natural way to genotype single-cells and derive clone assignments for downstream analysis.


## Table of Contents
- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [I/O Data Formats](#io-data-formats)
    - [Input Data](#input-data)
        - [Candidate tree format](#conipher-tree-format)
        - [Read counts format](#read-counts-file-format)
    - [Output Data](#output-files)
- [Usage](#usage)
    - [Arborist CLI tool](#arborist-cli-tool)
    - [`arborist` package](#arborist-package)


## Features
- Ranks candidate SNV phylogenies via estimation of a lower bound on the posterior probability
- Assigns cells to their most probable clones and computes the approximate posterior probability of assignment to each clone in the tree.
- Assigns SNVs to their most probable SNV clusters and computes the approximate posterior probability of assignment to each SNV cluster in the tree.
- Functions to compute the cell mutational burden (cmb) on the output of `arborist` for additional valdiation.


---


---

## Dependencies
The `arborist` package requires the following dependencies:
- `python>=3.7`
- `numba`
- `pandas`
- `numpy`
- `scipy`
- `pygrahviz`
- `networkx`



## Installation


Clone the respository and `sim-it` as a submodule (recommended). This will also clone the Sim-it repository.
Install packages locally. Note: dependencies will be installed automatically via pip.

```bash
git clone --recurse-submodules git@github.com:VanLoo-lab/Arborist.git
cd Arborist
pip install .
cd simulator
pip install .
```


Alternatively, just clone and install the Arborist repository from github.

```bash
git clone https://github.com/VanLoo-lab/Arborist.git
cd Arborist
pip install .

```


## I/O Data Formats

### Input Data

#### **Read Counts File Format**
The input read counts file should be in CSV format or `panda.DataFrame` and contain the following columns:
| Column | Description |
|---------|------------|
| `cell` | cell identifier |
| `snv` | SNV id |
| `total` | Total number of reads for cell and SNV |
| `alt` | Number of variant reads for cell and SNV |
| `cluster` | Cluster id to which the SNV belongs. |




#### **Tree Format**
- Trees are labeled numerically (e.g., `# .* tree 1 .*`).
- Each tree consists of comma seperated parent-child relationships.
- The root of each tree is always labeled by -1 and represents the normal clone. This is because the scDNA-seq may contain normal cells.


**Example:**
```
#candidate tree 0
0,4
2,1
1,0
-1,2
1,3
#candidate tree 1
2,4
4,0
2,3
-1,2
4,1
#candidate tree 2
4,0
2,1
3,4
-1,2
1,3
#candidate tree 3
4,0
2,1
1,4
2,3
-1,2
#candidate tree 4
2,4
4,0
2,1
2,3
-1,2
#candidate tree 5
2,4
2,1
1,0
-1,2
1,3
```



---

## Output Data
Example input and output files can be found in the `example/input` and `example/output` directories.
<!-- 
| File | Description |
|------|------------|
| `ranked_trees.csv` | Ranked trees with their probabilities. |
| `top20_trees_cell_assignment.csv` | Cell assignments for the top 20 ranked trees. |
| `Entropy.csv` | Entropy estimates for cell assignments. | -->


## Usage

### Arborist CLI tool
```bash
$ arborist -h
usage: arborist [-h] -R READ_COUNTS -T TREES [--alpha ALPHA] [--max-iter MAX_ITER] [--ranking RANKING] [--cell-assign CELL_ASSIGN] [--snv-assign SNV_ASSIGN] [--q_y Q_Y] [--q_z Q_Z] [-v] [-d DRAW] -t TREE
                [--prior PRIOR] [--map-assign] [--pickle PICKLE]

Arborist: a method to rank SNV clonal trees using scDNA-seq data.

options:
  -h, --help            show this help message and exit
  -R READ_COUNTS, --read_counts READ_COUNTS
                        Path to read counts CSV file with columns 'snv', 'cell', 'cluster', 'total', 'alt'
  -T TREES, --trees TREES
                        Path to file containing all candidate trees to be ranked.
  --alpha ALPHA         Per base sequencing error
  --max-iter MAX_ITER   max number of iterations.
  --ranking RANKING     Path to where tree ranking output should be saved.
  --cell-assign CELL_ASSIGN
                        Path to where cell assignments output should be saved.
  --snv-assign SNV_ASSIGN
                        Path to where snv assignments output should be saved.
  --q_y Q_Y             Path to where the approximate SNV posterior should be saved
  --q_z Q_Z             Path to where the approximate cell posterior should be saved
  -v, --verbose         Print verbose output
  -d DRAW, --draw DRAW  Path to save the tree image
  -t TREE, --tree TREE  Path to save the top ranked tree file.
  --prior PRIOR         prior (gamma) on input SNV cluster assignment
  --map-assign
  --pickle PICKLE       path to where all pickled tree fits should be saved.
```

#### Example
```bash
   arborist -R {read_count_fname} -T {trees_fname} \
   --alpha 0.001 --ranking {output_tree_rank_fname} \
   --draw {output_png_or_dot_fname} \
   --cell-assign {output_cell_assignment_fname} --verbose
```

### CMB CLI tool usage
```
cmb -h                                                                                        
usage: cmb [-h] -R READ_COUNTS -T TREES [--cell-assign CELL_ASSIGN] [--sapling] [--min-cells MIN_CELLS] [-o OUT] [--conipher]

options:
  -h, --help            show this help message and exit
  -R READ_COUNTS, --read_counts READ_COUNTS
                        Path to read counts CSV file with columns 'cell','snv', 'cluster', 'total', 'alt'
  -T TREES, --trees TREES
                        Path to tree file
  --cell-assign CELL_ASSIGN
                        Path to where cell assignments, with columns 'cell', 'snv', 'tree', 'clone'
  --sapling             Use sapling format for tree edges, otherwise conipher format is assumed.
  --min-cells MIN_CELLS
                        minimum number of cells to compute the scores for a clade
  -o OUT, --out OUT     filename where cmb values will be saved
  --conipher            if the tree is in conipher format

```

#### Example
```bash
   cmb -R {read_count_fname} -T {trees_fname} \
   --cell-assign {output_cell_assignment_fname} \
   --min-cells 20 -o {output_cmb_fname}
```

### `arborist` Package

Run `arborist` on a set of candidate trees and read counts. The below example generates simulated data using the `sim-it` simulator 
and then uses `arborist` to rank the trees. 

```python
import simit.simulator as sim
from arborist.arborist import rank_trees


gt, simdata = sim.simulate(
    seed=34,
    num_cells=1000,
    num_snvs=5000,
    coverage=0.02,
    nclusters=10,
    candidate_trees=None,
    candidate_set_size=10,
    cluster_error_prob=0.25,
    min_proportion=0.05
)

print(f"Candidate set size: {len(simdata.candidate_set)}")


rankings, best_fit, all_tree_fits = rank_trees(
    simdata.candidate_set,
    simdata.read_counts,
    alpha=0.001,
    max_iter=10,
    tolerance=1,
    verbose=True,
    prior=0.7,
)

print(rankings.head())


```

Perform validation on the trees using the cell mutational burden `cmb` metric.
```python
from arborist.cmb import cmb

#read a mapping of cluster to a list of SNVs from the read_counts
psi = dat[["snv", "cluster"]].drop_duplicates()
psi = dict(zip(psi["snv"], psi["cluster"]))
clade_snvs = defaultdict(list)
for j, cluster in psi.items():
   clade_snvs[cluster].append(j)

#minumum number of cells within/outside of clade needed to compute the metric.
min_cells = 20

#compute the cmb of each tree, clade and cell in the arborist output dataframe cell_assign
cmb_df = cmb(cell_assign,trees,clade_snvs, read_counts, min_cells)

print(cmb_df.head())
```



