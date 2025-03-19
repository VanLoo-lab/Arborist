import argparse
import pandas as pd
import numpy as np
from scipy.stats import binom
from scipy.special import logsumexp
from collections import defaultdict
from scipy.stats import entropy
import pygraphviz as pgv
from .utils import read_tree_edges_conipher, read_tree_edges_sapling, visualize_tree


def parse_arguments():
    parser = argparse.ArgumentParser(description="Tree ranking script")
    parser.add_argument(
        "-R",
        "--read_counts",
        required=True,
        help="Path to read counts CSV file with columns 'cell', 'cluster', 'total', 'alt'",
    )
    parser.add_argument("-T", "--trees", required=True, help="Path to tree file")
    parser.add_argument(
        "--sapling",
        action="store_true",
        help="Use sapling format for tree edges, otherwise conipher format is assumed.",
    )
    parser.add_argument(
        "--alpha",
        required=False,
        type=float,
        default=0.001,
        help="Per base sequencing error",
    )
    parser.add_argument(
        "--topn",
        required=False,
        type=int,
        default=25,
        help="Filter only the top n trees, default is 25.",
    )
    parser.add_argument(
        "--ranking",
        required=False,
        type=str,
        help="Path to where tree ranking output should be saved",
    )
    parser.add_argument(
        "--cell-assign",
        required=False,
        type=str,
        help="Path to where cell assignments output should be saved",
    )
    parser.add_argument(
        "-v", "--verbose", help="Print verbose output", action="store_true"
    )
    parser.add_argument(
        "-d", "--draw", required=False, type=str, help="Path to save the tree image"
    )

    return parser.parse_args()

def run():
    raise NotImplementedError
    #initialize snv clusters 
    #create the tree 
    prev_likelihood = np.inf
    while True:
        tree_like = cell_assign = find_cell_assignments()
        tree_like, snv_clusters = find_snv_clusters()

        if np.abs(prev_likelihood - tree_like) <  0.01:
            break
    
    return tree_like, cell_assign, snv_clusters


def find_snv_clusters(read_counts, tree, cell_assignment):
    raise NotImplementedError

def find_cell_assignments(
    read_counts, tree, error_rate=0.001
):
    """
    Processes read counts and calculates probabilities for cell-to-clone assignments
    based on a given evolutionary tree and error rate.

    Parameters
    ----------
    read_counts : pandas.DataFrame
        A DataFrame containing read count data with the following columns:
        - 'cell': Identifier for the cell.
        - 'cluster': Identifier for the cluster (clone).
        - 'total': Total number of reads.
        - 'alt': Number of alternate reads.
    tree : list of tuple
        A list of tuples representing the evolutionary tree. Each tuple is of the form
        (parent, child), where `parent` and `child` are cluster identifiers.
    error_rate : float, optional
        The sequencing error rate, by default 0.001.

    Returns
    -------
    tuple
        A tuple containing:
        - product : float
            The sum of the maximum log-likelihoods for each cell.
        - Cell_assignment : pandas.DataFrame
            A DataFrame where rows correspond to cells and columns correspond to clones.
            Each entry represents the log-likelihood of the cell being assigned to the clone.

    Notes
    -----
    - The function calculates the likelihood of each cell being assigned to each clone
      based on the evolutionary tree and read counts.
    - The probabilities for mutation and non-mutation are derived from the error rate.
    - The binomial log probability mass function is used to compute log-likelihoods.

    Examples
    --------
    >>> read_counts = pd.DataFrame({
    ...     'cell': ['cell1', 'cell1', 'cell2'],
    ...     'cluster': [1, 2, 1],
    ...     'total': [100, 150, 120],
    ...     'alt': [10, 15, 12]
    ... })
    >>> tree = [(0, 1), (1, 2)]
    >>> product, Cell_assignment = process_read_counts_and_calculate_probabilities(read_counts, tree)
    >>> print(product)
    >>> print(Cell_assignment)
    """

    child_to_parent = {child: parent for parent, child in tree}
    all_clones = set(child_to_parent.keys()).union(set(child_to_parent.values()))

    def get_ancestors(child):
        ancestors = []
        while child in child_to_parent:
            ancestors.append(child)
            child = child_to_parent[child]
        ancestors.append(child)
        return ",".join(map(str, ancestors[::-1]))

    evolution_matrix = pd.DataFrame(
        [
            (clone, get_ancestors(clone) if clone in child_to_parent else str(clone))
            for clone in all_clones
        ],
        columns=["Child", "Ancestors"],
    )

    # filter out any clusters that are not in the tree.
    filtered_read_counts = read_counts[
        read_counts["cluster"].isin(evolution_matrix["Child"])
    ]
    cell_reads = defaultdict(lambda: defaultdict(list))
    for _, row in filtered_read_counts.iterrows():
        cell_reads[row["cell"]][row["cluster"]].append((row["total"], row["alt"]))

    clones = evolution_matrix["Child"].values
    cells = list(cell_reads.keys())
    Cell_assignment = pd.DataFrame(index=cells, columns=clones, dtype=float)

    p_mutated = 0.5 - error_rate + 0.5 * error_rate / 3
    p_not_mutated = error_rate / 3

    for cell in cells:
        log_likelihoods = np.zeros(len(clones))
        reads = cell_reads[cell]

        for clone_idx, clone in enumerate(clones):
            ancestor_value = evolution_matrix.loc[
                evolution_matrix["Child"] == clone, "Ancestors"
            ].values[0]
            cluster_numbers = set(map(int, ancestor_value.split(",")))

            total_reads_arr, variant_reads_arr, p_arr = [], [], []
            for cluster, read_data in reads.items():
                data = np.array(read_data, dtype=int)
                total_reads, variant_reads = data[:, 0], data[:, 1]
                p = p_mutated if int(cluster) in cluster_numbers else p_not_mutated
                total_reads_arr.extend(total_reads)
                variant_reads_arr.extend(variant_reads)
                p_arr.extend([p] * len(total_reads))

            if total_reads_arr:
                log_likelihoods[clone_idx] = np.sum(
                    binom.logpmf(variant_reads_arr, total_reads_arr, p_arr)
                )

        Cell_assignment.loc[cell, :] = log_likelihoods

    product = np.sum(Cell_assignment.max(axis=1))
    return product, Cell_assignment


def rank_trees(tree_list, read_counts, alpha=0.001, topn=None, verbose=False):
    """
    Rank SNV phylogenetic trees based on their likelihood given scDNA-seq read count data and calculate entropy for cell assignments.
    This function processes a list of phylogenetic trees and their associated read counts to calculate
    the likelihood of each tree. After ranking, it selects the top `topn` trees,
    and computes entropy for cell assignments based on the selected trees.
    Parameters
    ----------
    tree_list : list
        A list of phylogenetic trees to be ranked.
    read_counts : pandas.DataFrame
        A DataFrame containing read counts for each cell.
    alpha : float, optional
        The per-base sequencing error rate for computing the likelihood of the tree given the read counts (default is 0.001).
    topn : int, optional
        The number of top-ranked trees to return. If None, all trees are returned (default is None).
    Returns
    -------
    ranked_trees : pandas.DataFrame
        A DataFrame containing the top-ranked trees with their likelihoods and posterior probabilities.
    filtered_assignments : pandas.DataFrame
        A DataFrame containing cell assignments for the top-ranked trees, including clone and tree information.
    Entropy : pandas.DataFrame
        A DataFrame containing entropy values for each cell, along with associated tree and clone information.
    Notes
    -----
    - The cell assignment posterior probabilities are normalized using the log-sum-exp to avoid numerical instability.
    - Entropy is calculated for each cell and each tree based on the posterior probabilities of assignment to each clones.
    Examples
    --------
    >>> ranked_trees, filtered_assignments, Entropy = rank_trees(tree_list, read_counts, alpha=0.001, topn=25)
    >>> print(ranked_trees.head())
    >>> print(filtered_assignments.head())
    >>> print(Entropy.head())
    """

    tree_probabilities = []
    combined_outputs = []

    for idx, tree in enumerate(tree_list):

        raw_probability, Cell_assignment_df = (
            find_cell_assignments(read_counts, tree, alpha)
        )

        # Extract actual node labels from the tree (not "Clone_X" format)
        clone_labels = Cell_assignment_df.columns.tolist()
        assigned_clones = Cell_assignment_df.select_dtypes(include=[np.number]).idxmax(
            axis=1
        )

        clone_labels = [f"clone_{x}_posterior" for x in clone_labels]
        Cell_assignment_df.columns = clone_labels

        # Assign each cell to the most probable clone
        Cell_assignment_df["clone"] = assigned_clones
        Cell_assignment_df["tree"] = idx  # Use direct tree index (0, 1, …)
        Cell_assignment_df["likelihood"] = raw_probability
        combined_outputs.append(Cell_assignment_df.reset_index())

        tree_probabilities.append({"tree": idx, "likelihood": raw_probability})

        if verbose:
            print(f"Processed tree {idx} with likelihood {raw_probability}.")

    if verbose:
        print(f"Processed {len(tree_list)} trees.")
        print("Writing output to disk...")

    # Create a DataFrame of tree probabilities
    df = pd.DataFrame(tree_probabilities)

    # Normalize probabilities
    logsum = logsumexp(df["likelihood"])
    df["posterior"] = np.exp(df["likelihood"] - logsum) * 100

    # Rank trees by likelihood
    df.sort_values(by=["likelihood"], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Select top `topn` ranked trees
    ranked_trees = df.iloc[:topn]

    # Filter Cell_assignment_df to include only top-ranked trees
    filtered_assignments = pd.concat(combined_outputs, ignore_index=True)
    filtered_assignments = filtered_assignments[
        filtered_assignments["tree"].isin(ranked_trees["tree"])
    ]

    # Compute posterior probabilities
    posterior_probs = np.exp(
        filtered_assignments[clone_labels]
        - logsumexp(filtered_assignments[clone_labels], axis=1, keepdims=True)
    )

    # Compute entropy
    entropy_values = posterior_probs.apply(lambda row: entropy(row, base=2), axis=1)

    # Create a single output DataFrame
    cell_assignments = pd.DataFrame()
    cell_assignments["cell"] = filtered_assignments["index"]  # Use original cell labels
    cell_assignments["tree"] = filtered_assignments["tree"]  # Use tree index directly
    cell_assignments["clone"] = filtered_assignments[
        "clone"
    ]  # Use actual node label from the tree
    cell_assignments["entropy"] = entropy_values

    # Rename posterior probability columns to actual clone labels
    posterior_probs.columns = clone_labels
    cell_assignments = pd.concat([cell_assignments, posterior_probs], axis=1)

    return ranked_trees, cell_assignments


def main():
    args = parse_arguments()
    read_counts = pd.read_csv(args.read_counts)

    read_trees = read_tree_edges_conipher

    if args.sapling:
        read_trees = read_tree_edges_sapling

    candidate_trees = read_trees(args.trees)
    ranked_trees, cell_assignments = rank_trees(
        candidate_trees,
        read_counts,
        alpha=args.alpha,
        topn=args.topn,
        verbose=args.verbose,
    )

    tree_idx = ranked_trees["tree"].tolist()[0]
    if args.draw:

        best_tree = cell_assignments[cell_assignments["tree"] == tree_idx]
        cell_assign = dict(zip(best_tree["cell"], best_tree["clone"]))
        visualize_tree(
            candidate_trees[tree_idx],
            cell_assignment=cell_assign,
            output_file=args.draw,
        )

    # Save results
    if args.ranking:
        ranked_trees.to_csv(args.ranking, index=False)
    if args.cell_assign:
        cell_assignments.to_csv(args.cell_assign, index=False)
