import argparse
import pandas as pd
import numpy as np
import numba
from numba import njit, prange
from scipy.stats import binom
from collections import defaultdict
from .utils import read_tree_edges_conipher, read_tree_edges_sapling, visualize_tree
from scipy.special import softmax
from .treefit import TreeFit


numba.set_num_threads(10)



@njit
def logsumexp_inline(log_terms, n):
    max_log = -np.inf
    for i in range(n):
        if log_terms[i] > max_log:
            max_log = log_terms[i]
    sum_exp = 0.0
    for i in range(n):
        sum_exp += np.exp(log_terms[i] - max_log)
    return max_log + np.log(sum_exp)

@njit(parallel=True, cache=True)
def compute_q_z_sparse(cell_ptr, snv_idx, log_likes, log_q_y, presence, n_cells, n_clones):
    q_z = np.zeros((n_cells, n_clones))

    for i in prange(n_cells):
        for r in range(n_clones):
            acc = 0.0
            for k in range(cell_ptr[i], cell_ptr[i + 1]):
                log_present = log_likes[2 * k]
                log_absent = log_likes[2 * k + 1]
                j = snv_idx[k]

                log_terms = np.empty(n_clones)
                for p in range(n_clones):
                    log_prob = log_present * presence[p, r] + log_absent * (1 - presence[p, r])
                    log_terms[p] = log_q_y[j, p] + log_prob

                acc += logsumexp_inline(log_terms, n_clones)

            q_z[i, r] = acc

        # Normalize to softmax
        max_log = np.max(q_z[i])
        q_z[i] = np.exp(q_z[i] - max_log)
        q_z[i] /= np.sum(q_z[i])

    return q_z

@njit 
def log_array(arr):
    log_arr = np.full_like(arr, -np.inf)
    for j in range(arr.shape[0]):
        for p in range(arr.shape[1]):
            if arr[j, p] > 0.0:
                log_arr[j, p] = np.log(arr[j, p])
    return log_arr

@njit(parallel=True, cache=True)
def compute_q_y_sparse(snv_ptr, cell_idx, log_likes, log_q_z, presence, n_snvs, n_clones):
    q_y = np.zeros((n_snvs, n_clones))

    for j in prange(n_snvs):
        for p in range(n_clones):
            acc = 0.0
            for k in range(snv_ptr[j], snv_ptr[j+1]):
                log_present = log_likes[2 * k]
                log_absent = log_likes[2 * k + 1]
                i = cell_idx[k]
                # Build log-sum-exp terms for this (i, j) pair
                log_terms = np.empty(n_clones)
                for r in range(n_clones):
                
                        log_prob = log_present * presence[p, r] + log_absent * (1 - presence[p, r])
                        log_terms[r] = log_q_z[i,r] + log_prob
                   

     
                acc += logsumexp_inline(log_terms, n_clones)

            q_y[j, p] = acc

        # Normalize to softmax
        max_log = np.max(q_y[j])
        q_y[j] = np.exp(q_y[j] - max_log)
        q_y[j] /= np.sum(q_y[j])

    return q_y

@njit
def compute_likelihood_sparse(cell_idx, snv_idx, log_likes, log_q_y, loq_q_z, presence, n_clones):

    expected_log_likelihood = 0.0
    for k in range(len(cell_idx)):
        i = cell_idx[k]
        j = snv_idx[k]
        log_present = log_likes[2 * k]
        log_absent = log_likes[2 * k + 1]

        for r in range(n_clones):
            for p in range(n_clones):
    
                    log_w = loq_q_z[i, r] + log_q_y[j, p]
                    log_prob = log_present * presence[p, r] + log_absent * (1 - presence[p, r])
                    expected_log_likelihood += np.exp(log_w) * log_prob

    return expected_log_likelihood 


def build_index_pointers(index_array, n):
    """
    Builds a pointer array such that data[ptr[i]:ptr[i+1]] gives all entries for index i.
    Returns: pointer array and sorted index array
    """
    counts = np.zeros(n + 1, dtype=np.int32)
    for idx in index_array:
        counts[idx + 1] += 1
    for i in range(1, n + 1):
        counts[i] += counts[i - 1]
    sort_idx = np.argsort(index_array)
    return counts, sort_idx


def build_sparse_input(df, cell_to_idx, snv_to_idx):
    """
    Converts (cell, snv) log likelihood DataFrame into sparse COO-style NumPy arrays.
    Returns: (cell_idx, snv_idx, log_matrix) where log_matrix[:, 0] = log_present, log_matrix[:, 1] = log_absent
    """
    n = df.shape[0]
    cell_idx = np.zeros(n, dtype=np.int32)
    snv_idx = np.zeros(n, dtype=np.int32)
    log_matrix = np.zeros(2 * n, dtype=np.float64)

    for i, row in enumerate(df.itertuples(index=False)):
            cell_idx[i] = cell_to_idx[row.cell]
            snv_idx[i] = snv_to_idx[row.snv]
            log_matrix[2 * i] = row.log_present
            log_matrix[2 * i + 1] = row.log_absent

    return cell_idx, snv_idx, log_matrix




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
    parser.add_argument(
         "--edge-sep", required=False, type=str,  default=" ", help="edge separator in tree list"
    )

    return parser.parse_args()


def enumerate_presence(genotype_matrix: pd.DataFrame, clones: list) -> np.array:
    """
    Enumerates the presence of each clone in the genotype matrix.
    """
    n_clones =len(clones)
    presence = np.zeros((n_clones, n_clones), dtype=int)
 
    for p in range(n_clones):
        clone_p = clones[p]
        for r in range(n_clones):
       
            clone_r = clones[r]
            # check if p is an ancestor of r
            p_descendant_set = genotype_matrix.loc[
                genotype_matrix["clone"] == clone_p, "descendants"
            ].values[0]
        
            if clone_r in p_descendant_set:
                presence[p,r] = 1
            else:
                presence[p,r] = 0
    return presence

def initialize_q_y(read_counts: pd.DataFrame,clones: list, snv_to_idx:dict) -> dict:
    """
    Initializes q(y_j = p) for each SNV j and clone p based on cluster assignment:
    """

  
    cluster = dict(zip(read_counts["snv"], read_counts["cluster"]))
    q_y = np.zeros((len(cluster), len(clones)), dtype=np.float64)
    for snv in cluster:
        j = snv_to_idx[snv]
        for p, clone_p in enumerate(clones):
            q_y[j,p]= 0.99 if cluster[snv] == clone_p else 0.01 / (len(clones) - 1)
    return q_y

    
def compute_likelihood(cell_snv_groups: dict, snvs_per_cell: dict, q_y: dict, q_z: dict, 
                       clones: list, presence: dict) -> float:
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
        rows = [cell_snv_groups[(cell, snv)] for snv in snvs_per_cell[cell] if (cell, snv) in cell_snv_groups]
        if not rows:
            continue
        snvs = [row["snv"] for row in rows]
        for r in clones:
            p_z = q_z[cell][r]
            for p in clones:
                terms = [
                    (row["log_present"] if presence[(p, r)] else row["log_absent"]) * p_z * q_y[snv][p]
                    for row, snv in zip(rows, snvs)
                ]
                expected_likelihood += np.sum(terms)
    return expected_likelihood
        

              

@njit
def run(presence: np.ndarray,
        log_like_matrix: np.ndarray,
        cell_idx: np.ndarray,
        snv_idx: np.ndarray,
        n_cells: int,
        n_snvs: int,
        n_clones: int,
        q_y: np.ndarray,
        cell_ptr: np.ndarray,
        snv_ptr: np.ndarray,
        max_iter=10,
        tolerance=1,
        verbose=False):

    best_likelihood = -np.inf
    best_q_z = np.zeros((n_cells, n_clones))
    best_q_y = np.zeros((n_snvs, n_clones))
    converged = False

    for it in range(max_iter):
        log_q_y = log_array(q_y)
        q_z = compute_q_z_sparse(cell_ptr,snv_idx, log_like_matrix, log_q_y, presence, n_cells, n_clones)
        log_q_z = log_array(q_z)

        q_y = compute_q_y_sparse(snv_ptr, cell_idx, log_like_matrix, log_q_z, presence, n_snvs, n_clones)
        likelihood = compute_likelihood_sparse(cell_idx, snv_idx, log_like_matrix, log_q_y, log_q_z, presence, n_clones)

        likelihood = 0.0  # You can optionally implement compute_likelihood_sparse

        if np.abs(likelihood - best_likelihood) < tolerance:
            if verbose:
                print(f"Converged after {it} iterations.")
                print(f"Best likelihood: {best_likelihood}")
            converged = True

        if likelihood > best_likelihood:
            best_q_z = q_z.copy()
            best_q_y = q_y.copy()
            best_likelihood = likelihood

        if converged:
            break

    return best_likelihood, best_q_z, best_q_y
        







def get_descendants_matrix(tree: list) -> pd.DataFrame:


    parent_to_child = defaultdict(list)
    for parent, child in tree:
        parent_to_child[parent].append(child)

    all_clones = set(parent_to_child.keys()).union(set(child for children in parent_to_child.values() for child in children))

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
        columns=["clone", "descendants"]
    )

    return evolution_matrix

def precompute_log_likelihoods(read_counts: pd.DataFrame, error_rate=0.001) -> pd.DataFrame:
    """
    Preprocesses the read counts by precomputing the log probabilities
    """

    read_counts["log_absent"] = binom.logpmf(
        read_counts["alt"], read_counts["total"], error_rate / 3
    )
    read_counts["log_present"] = binom.logpmf(
        read_counts["alt"], read_counts["total"], 0.5 - error_rate + 0.5 * error_rate / 3
    )

    cells = read_counts["cell"].unique() 
    snvs = read_counts["snv"].unique()

    cell_to_idx = {cell: idx for idx, cell in enumerate(cells)}
    snv_to_idx = {snv: idx for idx, snv in enumerate(snvs)}

    #create a dictionary of dictionaries
    #
   
    return read_counts, cell_to_idx, snv_to_idx

def tree_to_clone_set(tree: list) -> list: 
    clones = set()
    for u,v in tree:
        clones.add(u)
        clones.add(v)
    
    return clones


def rank_trees(tree_list: list, 
               read_counts: pd.DataFrame, 
               alpha: float=0.001,  
               max_iter: int=10,
               tolerance: float=5,
               verbose: bool=False) -> tuple:
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

    tree = tree_list[0]
    clone_set = tree_to_clone_set(tree) 
    for tree in tree_list:
        if tree_to_clone_set(tree) != clone_set:
            raise ValueError("All trees must have the same set of clones.")

    clones = list(clone_set)
    # Filter read_counts to only include cells and SNVs present in the tree
    read_counts = read_counts[
        read_counts["cluster"].isin(clones)
    ]





    #appends columns log_absent and log_present to read_counts
    read_counts, cell_to_idx, snv_to_idx = precompute_log_likelihoods(read_counts, alpha)
    q_y = initialize_q_y(read_counts, clones, snv_to_idx)
    cell_idx, snv_idx, log_like_matrix = build_sparse_input(read_counts, cell_to_idx, snv_to_idx)

    n_cells = len(cell_to_idx)
    n_snvs = len(snv_to_idx)
    n_clones = len(clones)
    cell_ptr, cell_sort_idx = build_index_pointers(cell_idx, n_cells)
    snv_ptr, snv_sort_idx = build_index_pointers(snv_idx, n_snvs)
    cell_idx = cell_idx[cell_sort_idx]
    snv_idx = snv_idx[cell_sort_idx]
    log_like_matrix = log_like_matrix[np.repeat(cell_sort_idx, 2)]

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

        if verbose:
            print(f"Starting tree {idx}...")
        
        genotype_matrix = get_descendants_matrix(tree)
  

        presence = enumerate_presence(genotype_matrix, clones)

        #run doesn't know the labels, everything is in index space
        expected_log_like, q_z, q_y = run(presence, 
                log_like_matrix = log_like_matrix,
                cell_idx = cell_idx,
                snv_idx = snv_idx,
                n_cells = n_cells,
                n_snvs = n_snvs,
                n_clones = n_clones, 
                q_y = q_y,
                cell_ptr = cell_ptr,
                snv_ptr = snv_ptr,
                max_iter = max_iter,
                tolerance = tolerance,
                verbose = verbose)

        tfit = TreeFit(tree, idx, expected_log_like, q_z, q_y, cell_to_idx, snv_to_idx, clones)
        if verbose:
            print(f"Tree {idx} fit wtih likelihood: {expected_log_like}")
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

    candidate_trees = read_trees(args.trees, sep=args.edge_sep)

    likelihoods, tfit = rank_trees(
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

    likelihoods_df = pd.DataFrame.from_dict(likelihoods, orient="index", columns=["likelihood"])
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
