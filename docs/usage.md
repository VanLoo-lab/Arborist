# Arborist CLI tool
After installation, `arborist` can be run via the command line with the following usage:


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

## Examples
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