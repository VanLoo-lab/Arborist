import argparse
import pandas as pd
import numpy as np
from scipy.stats import binom
from scipy.special import logsumexp
from collections import defaultdict
from .utils import read_tree_edges_conipher, read_tree_edges_sapling, visualize_tree
from scipy.special import softmax
import itertools
from .treefit import TreeFit


def parse_arguments() -> argparse.Namespace:
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
        "--snv-assign",
        required=False,
        type=str,
        help="Path to where snv assignments output should be saved",
    )
    parser.add_argument(
        "--q_y",
        required=False,
        type=str,
        help="Path to where the SNV posterior should be saved",
    )
    parser.add_argument(
        "--q_z",
        required=False,
        type=str,
        help="Path to where the cell posterior should be saved",
    )
    parser.add_argument(
        "-v", "--verbose", help="Print verbose output", action="store_true"
    )
    parser.add_argument(
        "-d", "--draw", required=False, type=str, help="Path to save the tree image"
    )

    return parser.parse_args()


def enumerate_presence(genotype_matrix: pd.DataFrame, clones: list) -> dict:
    """
    Enumerates the presence of each clone in the genotype matrix.
    """

    presence = {}

    for p, r in itertools.product(clones, clones):

        # check if p is an ancestor of r
        p_descendant_set = genotype_matrix.loc[
            genotype_matrix["clone"] == p, "descendants"
        ].values[0]

        if r in p_descendant_set:
            presence[(p, r)] = True
        else:
            presence[(p, r)] = False
    return presence


def initialize_q_y(read_counts: pd.DataFrame, clones: list) -> dict:
    """
    Initializes q(y_j = p) for each SNV j and clone p based on cluster assignment:
    """

    q_y = {}
    cluster = dict(zip(read_counts["snv"], read_counts["cluster"]))
    for snv in read_counts["snv"].unique():
        q_y[snv] = {}
        for p in clones:
            q_y[snv][p] = 0.99 if cluster[snv] == p else 0.01 / (len(clones) - 1)
    return q_y


def compute_likelihood(
    cell_snv_groups: dict,
    snvs_per_cell: dict,
    q_y: dict,
    q_z: dict,
    clones: list,
    presence: dict,
) -> float:
    """
    Computes the log-likelihood of the data given the current assignments.

    Parameters
    ----------
    cell_snv_groups : dict
        Mapping of (cell, snv) -> row of read_counts
    snvs_per_cell : dict
        Mapping of cell -> set of SNVs observed in that cell
    q_y : dict
        Nested dict: q_y[snv][clone] = probability
    q_z : dict
        Nested dict: q_z[cell][clone] = probability
    clones : list
        List of clone identifiers.
    presence : dict
        Dictionary of (p, r) -> bool indicating if r is a descendant of p.
    Returns
    -------
    float
        The expected log-likelihood of the data given the current assignments.
    """
    expected_likelihood = 0.0
    for cell in snvs_per_cell:

        rows = [
            cell_snv_groups[(cell, snv)].iloc[0]
            for snv in snvs_per_cell[cell]
            if (cell, snv) in cell_snv_groups
        ]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        for r in clones:
            p_z = q_z[cell][r]
            for p in clones:
                col = "log_present" if presence[(p, r)] else "log_absent"
                probs = df[col].values
                for snv, prob in zip(df["snv"], df[col].values):

                    p_y = q_y[snv][p]
                    # if cell ==0 and snv == 'chr3:30521:T:G':
                    #     print(f"cell {cell}, snv {snv}, p {p}, log prob {prob}")

                    #     print(f"presence {presence[(p, r)]}, p_z {p_z}, p_y {p_y}")
                    #     print(f"log p_z: {np.log(p_z)} log p _y: {np.log(p_y)}")
                    #     print(f"q_z {q_z[cell]}, q_y {q_y[snv]}")

                    # p_ys = np.array([q_y[snv][p] for snv in df["snv"]])
                    terms = prob * np.exp(np.log(p_z) + np.log(p_y))
                    # print(f"cell {cell}, snv {snv}, r {r}, p {p}, p_z {p_z}, p_y {p_y}, terms {terms}")

                    expected_likelihood += terms
    return expected_likelihood


def run(
    tree: list,
    read_counts: pd.DataFrame,
    max_iter: int = 10,
    tolerance=1,
    verbose: bool = False,
) -> tuple:
    """
    Iteratively processes read counts and calculates probabilities for cell-to-clone assignments
    and SNV assignments based on a given evolutionary tree and error rate to get the overall tree likelihood.

    Parameters
    ----------
    tree : list of tuple
        A list of tuples representing the evolutionary tree. Each tuple is of the form
        (parent, child), where `parent` and `child` are cluster identifiers.
    read_counts : pandas.DataFrame
        A DataFrame containing read count data with the following columns:
        - 'cell': Identifier for the cell.
        - 'cluster': Identifier for the cluster (clone).
        - 'total': Total number of reads.
        - 'alt': Number of alternate reads.
    error_rate : float, optional
        The sequencing error rate, by default 0.001.
    verbose : bool, optional
        If True, print detailed information during processing, by default False.
    max_iter : int, optional
        Maximum number of iterations for the optimization loop, by default 10.
    min_rate : float, optional
        Minimum rate of change for convergence, by default 0.01.
    tree_idx : int, optional
        Index of the current tree being processed, by default None.
    Returns
    -------
    tuple
        A tuple containing:
        - product : float
            The sum of the maximum log-likelihoods for each cell.
        - Cell_assignment : pandas.DataFrame
            A DataFrame where rows correspond to cells and columns correspond to clones.
            Each entry represents the log-likelihood of the cell being assigned to the clone.
        - snv_assignments : pandas.DataFrame
            A DataFrame where rows correspond to SNVs and columns correspond to clusters.
            Each entry represents the log-likelihood of the SNV being assigned to the cluster.
    """

    # store best results

    best_likelihood = -np.inf
    best_q_z = None
    best_q_y = None
    converged = False

    genotype_matrix = get_descendants_matrix(tree)
    clones = genotype_matrix["clone"].unique()

    presence = enumerate_presence(genotype_matrix, clones)

    filtered_read_counts = read_counts[read_counts["cluster"].isin(clones)]

    # Pre-group by (cell, snv) to avoid repeated filtering
    cell_snv_groups = dict(tuple(filtered_read_counts.groupby(["cell", "snv"])))
    snvs_per_cell = defaultdict(set)
    for cell, snv in cell_snv_groups.keys():
        snvs_per_cell[cell].add(snv)
    cells_per_snv = defaultdict(set)
    for cell, snv in cell_snv_groups.keys():
        cells_per_snv[snv].add(cell)

    # initialize SNV posterior
    q_y = initialize_q_y(filtered_read_counts, clones)

    for it in range(max_iter):

        q_z = compute_q_z(cell_snv_groups, presence, clones, q_y, snvs_per_cell)

        q_y = compute_q_y(cell_snv_groups, presence, clones, q_z, cells_per_snv)

        likelihood = compute_likelihood(
            cell_snv_groups, snvs_per_cell, q_y, q_z, clones, presence
        )

        if verbose:
            print(f"Iteration {it}:")
            print(f"Likelihood: {likelihood}")

        if np.abs(likelihood - best_likelihood) < tolerance:
            if verbose:
                print(f"Converged after {it} iterations.")
                print(f"Best likelihood: {best_likelihood}")
            converged = True

        if likelihood > best_likelihood:

            best_q_z = q_z
            best_q_y = q_y
            best_likelihood = likelihood

        if converged:
            break

    return best_likelihood, best_q_z, best_q_y


def compute_q_z(
    cell_snv_groups: dict, presence: dict, clones: list, q_y: dict, snvs_per_cell: dict
) -> dict:
    """
    Computes q(z_i = r) ∝ ∏_j sum_p q(y_j=p) * P(a_ij | z_i=r, y_j=p)
    using log-sum-exp for numerical stability.

    Parameters
    ----------
    read_counts : pd.DataFrame
        Must include columns: 'cell', 'snv', 'log_present', 'log_absent', 'cluster'.
    genotype_matrix : pd.DataFrame
        Must include column 'Child' listing clone IDs.
    q_y : dict
        Nested dictionary: q_y[snv][cluster] = probability

    Returns
    -------
    q_z : dict
        Nested dict: q_z[cell][clone] = log-prob (unnormalized)
    """

    # Filter only valid clusters present in the tree

    q_z = defaultdict(dict)

    for i, snvs in snvs_per_cell.items():
        q_z_i_r = np.zeros(len(clones))
        for idx_r, r in enumerate(clones):

            for snv in snvs:
                key = (i, snv)
                if key not in cell_snv_groups:
                    continue
                row = cell_snv_groups[key].iloc[0]  # should be just one row

                q_zi_clones = []
                for p in clones:
                    if q_y[snv][p] > 0:
                        col = "log_present" if presence[p, r] else "log_absent"
                        log_prob = row[col]
                        log_q_y = np.log(q_y[snv][p])
                        q_zi_clones.append(log_q_y + log_prob)

                q_z_i_r[idx_r] += logsumexp(q_zi_clones)

        q_z_i_probs = softmax(q_z_i_r, axis=0)

        for idx_r, r in enumerate(clones):
            q_z[i][r] = q_z_i_probs[idx_r]

    return q_z


def compute_q_y(
    cell_snv_groups: dict, presence: dict, clones: list, q_z: dict, cells_per_snv: dict
) -> dict:
    """
    Computes q(y_j = p) ∝ ∏_i sum_r q(z_i=r) * P(a_ij | z_i=r, y_j=p)
    using log-sum-exp for numerical stability.

    Parameters
    ----------
    read_counts : pd.DataFrame
        Must include columns: 'cell', 'snv', 'log_present', 'log_absent'
    presence : dict
        Dictionary of (p, r) -> bool indicating if r is a descendant of p.
    clones : list
        List of clone identifiers.
    q_z : dict
        Posterior assignment: q_z[cell][r] = P(z_i = r)
    cell_snv_groups : dict
        Mapping of (cell, snv) -> row of read_counts
    cells_per_snv : dict
        Mapping of snv -> set of cells that observe it

    Returns
    -------
    q_y : dict
        Nested dict: q_y[snv][clone] = probability
    """

    q_y = defaultdict(dict)

    for snv in cells_per_snv:
        log_q_y_j_p = np.zeros(len(clones))

        for idx_p, p in enumerate(clones):
            for cell in cells_per_snv[snv]:
                key = (cell, snv)
                if key not in cell_snv_groups:
                    continue

                row = cell_snv_groups[key].iloc[0]  # should be just one row

                log_terms = []
                for r in clones:
                    if q_z[cell][r] > 0:
                        col = "log_present" if presence[(p, r)] else "log_absent"
                        log_prob = row[col]

                        log_q_z = np.log(q_z[cell][r])
                        log_terms.append(log_q_z + log_prob)

                log_q_y_j_p[idx_p] += logsumexp(log_terms)

        # Normalize across p for each snv j
        q_y_j_probs = softmax(log_q_y_j_p)

        for idx_p, p in enumerate(clones):
            q_y[snv][p] = q_y_j_probs[idx_p]

    return q_y


def get_descendants_matrix(tree: list) -> pd.DataFrame:

    parent_to_child = defaultdict(list)
    for parent, child in tree:
        parent_to_child[parent].append(child)

    all_clones = set(parent_to_child.keys()).union(
        set(child for children in parent_to_child.values() for child in children)
    )

    def dfs(node, tree):
        descendants = []
        stack = [node]
        while stack:
            current = stack.pop()
            for child in tree.get(current, []):
                if child not in descendants:
                    descendants.append(child)
                    stack.append(child)
        return [node] + descendants

    evolution_matrix = pd.DataFrame(
        [(clone, dfs(clone, parent_to_child)) for clone in all_clones],
        columns=["clone", "descendants"],
    )

    return evolution_matrix


def precompute_log_likelihoods(
    read_counts: pd.DataFrame, error_rate=0.001
) -> pd.DataFrame:
    """
    Preprocesses the read counts by precomputing the log probabilities
    """

    read_counts["log_absent"] = binom.logpmf(
        read_counts["alt"], read_counts["total"], error_rate / 3
    )
    read_counts["log_present"] = binom.logpmf(
        read_counts["alt"],
        read_counts["total"],
        0.5 - error_rate + 0.5 * error_rate / 3,
    )

    return read_counts


def rank_trees_pandas(
    tree_list: list,
    read_counts: pd.DataFrame,
    alpha: float = 0.001,
    topn: int = None,
    max_iter: int = 10,
    tolerance: float = 5,
    verbose: bool = False,
) -> tuple:
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

    # appends columns log_absent and log_present to read_counts
    read_counts = precompute_log_likelihoods(read_counts, alpha)
    best_likelihood = -np.inf
    likelihoods = {}
    for idx, tree in enumerate(tree_list):

        """
        # Main function to find cell assignments
        raw_probability, Cell_assignment_df = (
            run(genotype_matrix, read_counts, alpha, verbose)
            find_cell_assignments(read_counts, genotype_matrix, alpha)
        )
        """

        expected_log_like, q_z, q_y = run(
            tree=tree,
            read_counts=read_counts,
            max_iter=max_iter,
            tolerance=tolerance,
            verbose=verbose,
        )
        tfit = TreeFit(tree, idx, expected_log_like, q_z, q_y, {}, {}, [])
        likelihoods[idx] = expected_log_like
        if expected_log_like > best_likelihood:
            best_fit = tfit
            best_likelihood = expected_log_like
    return likelihoods, best_fit


def main():
    args = parse_arguments()
    read_counts = pd.read_csv(args.read_counts)

    read_trees = read_tree_edges_conipher

    if args.sapling:
        read_trees = read_tree_edges_sapling

    candidate_trees = read_trees(args.trees)

    likelihoods, tfit = rank_trees_pandas(
        candidate_trees,
        read_counts,
        alpha=args.alpha,
        topn=args.topn,
        verbose=args.verbose,
    )

    if args.draw:

        visualize_tree(
            tfit.tree,
            # cell_assign = cell_assign
            output_file=args.draw,
        )

    cell_assign = tfit.map_assign_z()
    snv_assign = tfit.map_assign_y()

    likelihoods_df = pd.DataFrame.from_dict(
        likelihoods, orient="index", columns=["likelihood"]
    )
    likelihoods_df = likelihoods_df.reset_index().rename(columns={"index": "tree_idx"})
    likelihoods_df = likelihoods_df.sort_values(by="likelihood", ascending=False)

    # # Save results
    if args.ranking:
        likelihoods_df.to_csv(args.ranking, index=False)

    if args.cell_assign:
        cell_assign.to_csv(args.cell_assign, index=False)
    if args.snv_assign:
        snv_assign.to_csv(args.snv_assign, index=False)
    if args.q_z:
        tfit.q_z_df().to_csv(args.q_z, index=False)
    if args.q_y:
        tfit.q_y_df().to_csv(args.q_y, index=False)
