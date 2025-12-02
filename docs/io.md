### Input Data
Example input files can be found in the [example/input](example/input) directory.

Arborist requires three inputs as three separate files:  
1.[a CSV file containing single-cell variant **A** and total **D** read count data](#read-counts-file-format)
2.[a CSV file containing the initial SNV clustering](#initial-snv-clustering-format)
3.[a text file containing the set of candidate clone trees](#tree-format)


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
- Each tree consists of delimited separated parent-child relationships. The `arborist` default delimiter is " " but different delimiters may be passed via the `--edge-delim` argument 
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
