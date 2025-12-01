# Overview
Arborist is a method to rank SNV clone trees inferred from bulk DNA sequencing data by leveraging low-pass single-cell DNA sequencing (scDNA-seq) data. The method is designed to prioritize the most probable tree, helping to resolve ambiguities in bulk tree inference. Arborist uses variational inference to compute the evidence lower bound on the posterior probability of each tree in the input candidate set and approximates the cell to clone assignment posterior distribution as well as the SNV to SNV cluster assignment posterior. Arborist not only helps to resolve tree ambiguity within the bulk solution space but also helps improve an SNV cluster and yields a natural way to genotype single-cells and derive cell-to-clone assignments for downstream analysis.


## Table of Contents
- [Dependencies](#dependencies)
- [Installation](#installation)
- [I/O Data Formats](#io-data-formats)
    - [Input Data](#input-data)
        - [Read counts format](#read-counts-file-format)
        - [Initial SNV clustering format](#initial-snv-clustering-file-format)
        - [Candidate tree format](#tree-file-format)
    - [Output Data](#output-files)
- [Usage](#usage)
    - [Arborist CLI tool](#arborist-cli-tool)
    - [Arborist Python Package](#arborist-package)
    <!-- - [`arborist` package](#arborist-package) -->
- [Example](#example)
<!-- 

## Features
- Ranks candidate SNV phylogenies via estimation of a lower bound on the posterior probability
- Assigns cells to their most probable clones and computes the approximate posterior probability of assignment to each clone in the tree.
- Assigns SNVs to their most probable SNV clusters and computes the approximate posterior probability of assignment to each SNV cluster in the tree.
- Functions to compute the cell mutational burden (cmb) on the output of `arborist` for additional valdiation. -->





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

Clone the respository and install the package using `pip`. *Note: dependencies will be installed automatically*


```bash
git clone https://github.com/VanLoo-lab/Arborist.git
cd Arborist
pip install .

```

To test the installation of the CLI tool and Python package, run the following:
```bash
arborist -h
python -c "from arborist import arborist"
```

## I/O Data Formats

### Input Data
Example input files can be found in the [example/input](example/input) directory.

Arborist requires three inputs as three seperate files:
1. [a CSV file containing single-cell variant $A$ and total $D$ read count data](#read-counts-file-format)
2. [a CSV file containing the initial SNV clustering](#initial-snv-clustering-format)
3. [a text file contained the set of candidate clone trees](#tree-format)


#### **Read Counts File Format**
The input read counts file should be in CSV format contain the following columns with header row in any order:
| Column | Description |
|---------|------------|
| `cell` | cell identifier, str or int |
| `snv` | SNV identifier, str or int |
| `total` | Total number of reads for cell and SNV |
| `alt` | Number of variant reads for cell and SNV |


**Example**
```
snv,cell,alt,total
0,4,0,1
0,5,0,1
0,29,0,1
0,45,0,1
0,48,0,1
0,64,0,1
0,76,0,1
0,92,0,1
0,97,0,1
```

### **Initial SNV Clustering File Format**
The initial SNV clustering file should be in CSV format and contain no headers. The order of the columns matter with the first column being the SNV identifier followed by the initial SNV cluster label. SNV identifiers should be consistent with the read count file and cluster labels should be consistent with the node labels in the candidate tree. Any SNV assigned to an SNV cluster label that is not present in the tree with be initialized with a uniform prior over all SNV clusters. 

| Column | Description |
|---------|------------|
| `snv` | SNV identifier, str or int |
| `cluster` | SNV cluster label, int |

**Example**
```
108,2
123,2
176,2
289,2
452,2
597,2
857,2
890,2
909,2
918,2
```


#### **Tree File Format**
- The header for each tree, i.e., '# Tree 1'  must contain '#' but the remaining text is unimportant
- Each tree consists of delimited seperated parent-child relationships. The `arborist` default delimiter is " " but different delimiters maye be passed via the `--edge-delim` argument 
- If clone trees do not contain a normal clone (this is common), then the argument `--add-normal` should be used to tell `arborist` to append a normal clone to the root of each clone tree. Any cells assigned to the root will be normal cells. 
- Node identifiers must be integers


**Example:**
```
    # Tree 1
    1 2
    1 3
    # Tree 2
    4 5
    4 6
```




---

## Output Data
Some example output files are located in the [example/output](example/output) directory.
Below is table describing the optional output files from `arborist`

| argument | Description |
|------|------------|
| `--pickle`| A pickled dictionary with tree index as key and `TreeFit` objects containing the fit for each clone tree|
| `--draw`| A visualization of the `arborist` top ranked clone tree |
| `--tree` | A flat text file similar to the input format containing the `arborist` top ranked clone tree |
| `--ranking` | a CSV file containg the ranking of the clone trees by the ELBO from best to worst. 'tree_idx' is the order listed in the candidate set |
| `--cell-assign` | a CSV file containing the MAP assignment of cell (id) to clone (assignment) |
| `--snv-assign` | a CSV file containing the MAP assignment of SNV (id) to cluster (assignment) |
| `--q_z` | a CSV file containing the approximate posterior distribution over cell-to-clone labels |
| `--q_y` | a CSV file containing the approximate posterior distribution over SNV-to-cluster labels |





## Usage
After installation, `arborist` can be run via the command line with usage as shown below. 

### Arborist CLI tool
```bash
$ arborist -h                                                        
usage: arborist [-h] -R READ_COUNTS -Y SNV_CLUSTERS -T TREES [--edge-delim EDGE_DELIM] [--add-normal] [--alpha ALPHA] [--max-iter MAX_ITER] [--prior PRIOR]
                [--pickle PICKLE] [-d DRAW] [-t TREE] [--ranking RANKING] [--cell-assign CELL_ASSIGN] [--snv-assign SNV_ASSIGN] [--q_y Q_Y] [--q_z Q_Z]
                [-j THREADS] [-v]

Arborist: a method to rank SNV clonal trees using scDNA-seq data.

options:
  -h, --help            show this help message and exit
  -R READ_COUNTS, --read_counts READ_COUNTS
                        Path to read counts CSV file with columns 'snv', 'cell', 'total', 'alt'
  -Y SNV_CLUSTERS, --snv-clusters SNV_CLUSTERS
                        Path to SNV clusters CSV file with unlabeled columns 'snv', 'cluster'. The order of columns matters
  -T TREES, --trees TREES
                        Path to file containing all candidate trees to be ranked.
  --edge-delim EDGE_DELIM
                        edge delimiter in candidate tree file.
  --add-normal          flag to add a normal clone if input trees do not already contain them
  --alpha ALPHA         Per base sequencing error
  --max-iter MAX_ITER   max number of iterations
  --prior PRIOR         prior (gamma) on input SNV cluster assignment
  --pickle PICKLE       path to where all pickled tree fits should be saved.
  -d DRAW, --draw DRAW  Path to where the tree visualization should be saved
  -t TREE, --tree TREE  Path to save the top ranked tree as a txt file.
  --ranking RANKING     Path to where tree ranking output should be saved
  --cell-assign CELL_ASSIGN
                        Path to where the MAP cell-to-clone labels should be saved
  --snv-assign SNV_ASSIGN
                        Path to where the MAP SNV-to-cluster labels should be saved.
  --q_y Q_Y             Path to where the approximate SNV posterior should be saved
  --q_z Q_Z             Path to where the approximate cell posterior should be saved
  -j THREADS, --threads THREADS
                        Number of threads to use
  -v, --verbose         Print verbose output
```

#### Example
The following is a minimal `arborist` example with default parameters and no output files to test the installation.
```bash
   arborist -R example/input/read_counts.csv \
   -T example/input/candidate_trees.txt \
   -Y example/input/input_clustering.csv 
```

If everything installed correctly you should see the following printout:
```
-----------Arborist complete!-----------

Tree index 0
ELBO: -311397.27
 2->1
 2->3
 2->4
 2->5
 3->6
 0->2
```

This example demonstrates how to modify parameters and write relevant output files. 
```bash
   arborist -R example/input/read_counts.csv \
   -T example/input/candidate_trees.txt \
   -Y example/input/input_clustering.csv \
   --prior 0.7  \
   --alpha 0.001 \
   --ranking example/output/tree_rankings.csv \
   --draw example/output/best_tree.png \
   --cell-assign example/output/cell_to_clone_labels.csv \
   --snv-assign example/output/snv_to_cluster_labels.csv 
```


## Arborist package

Arborist is also a Python package. Below is an example of how to use it.

```python
import pandas as pd 
from arborist import read_trees, arborist 

#read the trees in as edge lists 
candidate_trees = read_trees("example/input/candidate_trees.txt")
print(f"Candidate set size: {len(candidate_trees)}")
# Candidate set size: 30

read_counts = pd.read_csv("example/input/read_counts.csv")
read_counts.head()
"""
>>> read_counts.head()
   snv  cell  alt  total
0    0     4    0      1
1    0     5    0      1
2    0    29    0      1
3    0    45    0      1
"""

snv_clusters = pd.read_csv("example/input/input_clustering.csv", header=None, names=["snv", "cluster"]) 
snv_clusters.head()
"""
>>> snv_clusters.head()
   snv  cluster
0  108        2
1  123        2
2  176        2
3  289        2
4  452        2
"""

ranking, best_fit, all_fits =arborist(
    tree_list = candidate_trees,
    read_counts = read_counts,
    snv_clusters = snv_clusters,
    alpha = 0.001,
    max_iter = 10,
    tolerance = 1,
    gamma= 0.7,
    add_normal = False,
    threads = 10,
    verbose = False
)

print(best_fit)
"""
Tree 0
ELBO: -311397.27
 2->1
 2->3
 2->4
 2->5
 3->6
 0->2
"""

```

`TreeFit` objects can be used to obtain MAP assignments of cell-to-clone labels (`z`) or SNV-to-cluster labels (`y`) as well as explore the approximate posterior distributions (`qz`) and (`qy`)
```python

#TreeFit 
z = best_fit.map_assign_z()
z.head()
y = best_fit.map_assign_y()
y.head()

qz = best_fit.q_z_df()
qz.head()
qy = best_fit.q_y_df()
qy.head()


```
<!-- ### CMB CLI tool usage
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


 -->
