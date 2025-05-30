import argparse
import pandas as pd
import numpy as np
import numba
from numba import njit, prange
from scipy.stats import binom
from collections import defaultdict
from .utils import read_tree_edges_conipher, read_tree_edges_sapling, visualize_tree, get_descendants_matrix
from .treefit import TreeFit
import networkx as nx

numba.set_num_threads(12)


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
        "--beta",
        required=False,
        type=float,
        default=0.0,
        help="Regularization parameter for q(y) KL Divergence with initial distribution.",
    )
    parser.add_argument(
        "--max-iter",
        required=False,
        type=int,
        default=25,
        help="max number of iterations.",
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
    parser.add_argument("-t", "--tree", required=True, help="Path to save the top ranked tree file.")
    parser.add_argument(
         "--prior", required=False, type=float,  default=0.7, help="prior on input SNV cluster assignment"
    )

    return parser.parse_args()

@njit
def soft_max_with_annealing(arr, tau):
    for j in range(arr.shape[0]):
        max_log = np.max(arr[j])
        arr[j] = np.exp((arr[j] - max_log) / tau)
        arr[j] /= np.sum(arr[j])
    return arr

@njit
def soft_max(arr):
    for j in range(arr.shape[0]):
        max_log = np.max(arr[j])
        arr[j] = np.exp((arr[j] - max_log))
        arr[j] /= np.sum(arr[j])
    return arr


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

@njit(parallel=True)
def compute_q_z_sparse(cell_ptr, snv_idx, log_likes, log_q_y, presence, n_cells, n_clones, tau):
    q_z = np.zeros((n_cells, n_clones))

    for i in prange(n_cells):
        for r in range(n_clones):
            acc = 0.0
            for k in range(cell_ptr[i], cell_ptr[i + 1]):
                log_present = log_likes[2 * k]
                log_absent = log_likes[2 * k + 1]
                j = snv_idx[k]

                log_terms = np.empty(n_clones-1)
                for p in range(n_clones-1):
                    log_prob = log_present * presence[p, r] + log_absent * (1 - presence[p, r])
                    log_terms[p] = log_q_y[j, p] + log_prob

                acc += logsumexp_inline(log_terms, n_clones-1)

            q_z[i, r] = acc

        # Normalize to softmax
        # max_log = np.max(q_z[i])
        # q_z[i] = np.exp(q_z[i] - max_log)
        # q_z[i] /= np.sum(q_z[i])
    if tau > 0:
       q_z = soft_max_with_annealing(q_z, tau)
    else:
        q_z = soft_max(q_z)
    return q_z


@njit
def compute_kl_divergence(q_y, q_y_init):
    kl = 0.0
    for j in range(q_y.shape[0]):
        for p in range(q_y.shape[1]):
            # if q_y[j, p] > 0 and q_y_init[j, p] > 0:
                kl += q_y[j, p] * (np.log(q_y[j, p] + 1e-12) - np.log(q_y_init[j, p]+1e-12))
    return kl

@njit 
def log_array(arr):
    log_arr = np.full_like(arr, -np.inf)
    for j in range(arr.shape[0]):
        for p in range(arr.shape[1]):
            if arr[j, p] > 0.0:
                log_arr[j, p] = np.log(arr[j, p])
    return log_arr

@njit(parallel=True)
def compute_q_y_sparse(snv_ptr, cell_idx, log_likes, log_q_z, presence, n_snvs, n_clones, tau):
    q_y = np.zeros((n_snvs, n_clones-1))

    for j in prange(n_snvs):
        for p in range(n_clones-1):
            acc = 0.0
            for k in range(snv_ptr[j], snv_ptr[j + 1]):
                log_present = log_likes[2 * k]
                log_absent = log_likes[2 * k + 1]
                i = cell_idx[k]

                log_terms = np.empty(n_clones)
                for r in range(n_clones):
                    log_prob = log_present if presence[p, r] else log_absent
                    log_terms[r] = log_q_z[i, r] + log_prob

                acc += logsumexp_inline(log_terms, n_clones)

            q_y[j, p] = acc


        # # Normalize to softmax
        # max_log = np.max(q_y[j])
        # q_y[j] = np.exp(q_y[j] - max_log)
        # q_y[j] /= np.sum(q_y[j])
    
    # for j in range(q_y.shape[0]):
    #     for p in range(q_y.shape[1]):
    #         q_y[j,p] = q_y_init[j,p]*lamb + (1-lamb)*q_y[j,p]
    #     max_log = np.max(q_y[j])
    #     q_y[j] = np.exp(q_y[j] - max_log)
    #     q_y[j] /= np.sum(q_y[j])

    if tau > 0:
       q_y = soft_max_with_annealing(q_y, tau)
    else:
        q_y = soft_max(q_y)
    return q_y

@njit(parallel=True)
def compute_likelihood_sparse(cell_ptr, snv_idx, log_likes, log_q_y, log_q_z, presence, n_clones, n_cells):

    expected_log_likelihood = 0.0
    for i in prange(n_cells):
        for k in range(cell_ptr[i], cell_ptr[i + 1]):
            
            log_present = log_likes[2 * k]
            log_absent = log_likes[2 * k + 1]
            j = snv_idx[k]
            for r in range(n_clones):
                for p in range(n_clones-1):
                        # log_w = np.log(q_y[j,p]) + np.log(q_z[i,r])
                        log_w = log_q_z[i, r] + log_q_y[j, p]
                        log_prob = log_present if presence[p, r] else log_absent
                        expected_log_likelihood += np.exp(log_w) * log_prob

    return expected_log_likelihood 


def build_index_pointers(index_array, alt_idx, n, log_like_matrix):
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


    index_array = index_array[sort_idx]
    alt_idx = alt_idx[sort_idx]
    n_obs = sort_idx.shape[0]
    ll_mat = log_like_matrix.reshape(n_obs, 2)     # each row = [log_present, log_absent]
    ll_mat = ll_mat[sort_idx]                 # apply same permutation
    log_like_matrix = ll_mat.ravel() 
    return counts, index_array, alt_idx, log_like_matrix


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






def enumerate_presence(tree:list, clones:list, clusters:list) -> np.array:
    """
    Enumerates the presence of each clone in the genotype matrix.
    """
    T = nx.DiGraph(tree)
    n_clones =len(clones)
    n_clusters = len(clusters)
    presence = np.zeros((n_clusters, n_clones), dtype=int)
    clone_to_idx= {r: i for i,r in enumerate(clones)}
 
    for p, cluster_p in enumerate(clusters):
      
        desc = nx.descendants(T,cluster_p) | {cluster_p}
        for d in desc:
            presence[p, clone_to_idx[d]] = 1
     
    return presence


# def enumerate_presence(genotype_matrix: pd.DataFrame, clones: list) -> np.array:
#     """
#     Enumerates the presence of each clone in the genotype matrix.
#     """
#     n_clones =len(clones)
#     presence = np.zeros((n_clones, n_clones), dtype=int)
 
#     for p in range(n_clones):
#         clone_p = clones[p]
#         for r in range(n_clones):
       
#             clone_r = clones[r]
#             # check if p is an ancestor of r
#             p_descendant_set = genotype_matrix.loc[
#                 genotype_matrix["clone"] == clone_p, "descendants"
#             ].values[0]
        
#             if clone_r in p_descendant_set:
#                 presence[p,r] = 1
#             else:
#                 presence[p,r] = 0
#     return presence

def initialize_q_y(
    read_counts: pd.DataFrame,
    clusters: list,
    snv_to_idx: dict,
    prior: float
) -> np.ndarray:
    """
    Initialize the variational posterior q(y_j = p) for each SNV j and clone p.

    Parameters
    ----------
    read_counts : pd.DataFrame
        Must have columns 'snv' and 'cluster', where 'cluster' is the
        initial hard assignment ψ(snv) in [clones].
    clones : list
        List of clone identifiers (length k).
    snv_to_idx : dict
        Mapping from SNV identifiers to row indices 0..m-1 in the output array.

    Returns
    -------
    q_y : np.ndarray, shape (m, k)
        q_y[j, p] = 0.99 if ψ(j) == p else 0.01/(k-1).
    """
    # Build a quick map: snv -> its hard cluster
    cluster_map = dict(zip(read_counts["snv"], read_counts["cluster"]))
    m, k = len(cluster_map), len(clusters)
    q_y = np.zeros((m, k), dtype=np.float64)

    # Epsilon mass on correct cluster; uniform remainder

 
    correct =  prior
    other = (1- prior) / (k - 1) if k > 1 else 0.0

    for snv, assigned in cluster_map.items():
        j = snv_to_idx[snv]
        for p_idx, clone_id in enumerate(clusters):
            q_y[j, p_idx] = correct if (assigned == clone_id) else other

    return q_y

    

@njit
def init_score(presence: np.ndarray,
        log_like_matrix_cell_sort: np.ndarray,
        snv_idx: np.ndarray,
        n_cells: int,
        n_clones: int,
        q_y_init: np.ndarray,
        cell_ptr: np.ndarray,
        beta = 0.5):

    # Compute initial q_z and ELBO before any updates
    log_q_y = log_array(q_y_init)
    q_z = compute_q_z_sparse(cell_ptr, snv_idx, log_like_matrix_cell_sort, log_q_y, presence, n_cells, n_clones, tau=0)
    log_q_z = log_array(q_z)
    initial_likelihood = compute_likelihood_sparse(cell_ptr, snv_idx, log_like_matrix_cell_sort, log_q_y, log_q_z, presence, n_clones, n_cells)
    kl_penalty = compute_kl_divergence(q_y_init, q_y_init)
    initial_elbo = initial_likelihood - beta * kl_penalty
    return initial_elbo

              

@njit
def run(presence: np.ndarray,
        log_like_matrix_cell_sort: np.ndarray,
        log_like_matrix_snv_sort: np.ndarray,
        cell_idx: np.ndarray,
        snv_idx: np.ndarray,
        n_cells: int,
        n_snvs: int,
        n_clones: int,
        q_y_init: np.ndarray,
        cell_ptr: np.ndarray,
        snv_ptr: np.ndarray,
        max_iter=10,
        tolerance=1,
        beta = 0.5,
        verbose=False):

    tau_init = 1.0
    tau_final = 0.1
    # Compute initial q_z and ELBO before any updates
    log_q_y = log_array(q_y_init)
    q_z = compute_q_z_sparse(cell_ptr, snv_idx, log_like_matrix_cell_sort, log_q_y, presence, n_cells, n_clones, tau=0)
    log_q_z = log_array(q_z)
    initial_likelihood = compute_likelihood_sparse(cell_ptr, snv_idx, log_like_matrix_cell_sort, log_q_y, log_q_z, presence, n_clones, n_cells)
    kl_penalty = compute_kl_divergence(q_y_init, q_y_init)
    initial_elbo = initial_likelihood - beta * kl_penalty

    # Initialize best values to initial state
    best_likelihood = initial_elbo
    best_q_z = q_z.copy()
    best_q_y = q_y_init.copy()

    converged = False
   
    # psi = q_y.argmax(axis=1)
    # print(f"Initial unique clusters: {np.unique(psi).shape[0]}")
    # print(psi.shape)

    for it in range(max_iter):
        tau = tau_init * (tau_final / tau_init) ** (it / max_iter)

        q_y = compute_q_y_sparse(snv_ptr, cell_idx, log_like_matrix_snv_sort, log_q_z, presence, n_snvs, n_clones, tau=0)
        new_psi = q_y.argmax(axis=1)
        # num_different  = (new_psi !=psi).sum()
        # perc_different = num_different/psi.shape[0]
        # print(f"Number different: {num_different}, {100*perc_different}%")
        # print(f"Unique SNV clusters: {np.unique(psi).shape[0]}")
        log_q_y = log_array(q_y)
       
        q_z = compute_q_z_sparse(cell_ptr,snv_idx, log_like_matrix_cell_sort, log_q_y, presence, n_cells, n_clones, tau=0)

        log_q_z = log_array(q_z)


  
        likelihood = compute_likelihood_sparse(cell_ptr, snv_idx, log_like_matrix_cell_sort, log_q_y, log_q_z, presence, n_clones, n_cells)
        kl_penalty = compute_kl_divergence(q_y, q_y_init)
        # print(f"Likelihood: {likelihood:.2f}, KL: {kl_penalty:.2f}, beta*KL: {beta*kl_penalty:.2f}")
        likelihood = likelihood - beta*kl_penalty

        # # print(f"Likelihood: {likelihood:.2f}")
        # if verbose:
        #     print(f"Iteration {it}----------")
        #     print(f"Likelihood: {likelihood}")
    

        if np.abs(likelihood - best_likelihood) < tolerance:
            # if verbose:
            #     print(f"Converged after {it} iterations.")
            #     print(f"Best likelihood: {best_likelihood}")
            converged = True

        if likelihood > best_likelihood:
            best_q_z = q_z.copy()
            best_q_y = q_y.copy()
            best_likelihood = likelihood

        if converged:
            break

    return best_likelihood, best_q_z, best_q_y
        









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
               beta: float= 0,
               max_iter: int=10,
               tolerance: float=5,
               prune = False,
               topn = 5,
               prior=0.7,
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
    clusters = [c for c in clones if c != -1]
    # Filter read_counts to only include cells and SNVs present in the tree
    read_counts = read_counts[
        read_counts["cluster"].isin(clones)
    ]


    alt_sum = read_counts.groupby("snv")["alt"].sum()
    valid_snvs = alt_sum[alt_sum > 0].index
    read_counts = read_counts[read_counts["snv"].isin(valid_snvs)]


    #appends columns log_absent and log_present to read_counts
    read_counts, cell_to_idx, snv_to_idx = precompute_log_likelihoods(read_counts, alpha)
    q_y_init = initialize_q_y(read_counts, clusters, snv_to_idx, prior)
    cell_idx, snv_idx, log_like_matrix = build_sparse_input(read_counts, cell_to_idx, snv_to_idx)

    n_cells = len(cell_to_idx)
    n_snvs = len(snv_to_idx)
    n_clones = len(clones)
    # cell_ptr, cell_sort_idx = build_index_pointers(cell_idx, snv_idx, n_cells)
    cell_ptr, cell_idx_sort, snv_index_cell_sort, log_like_matrix_cell_sort = build_index_pointers(cell_idx, snv_idx, n_cells, log_like_matrix=log_like_matrix)
    snv_ptr, snv_idx_sort, cell_index_snv_sort, log_like_matrix_snv_sort = build_index_pointers( snv_idx, cell_idx, n_snvs, log_like_matrix=log_like_matrix)

                # flatten back into 1D

    best_likelihood = -np.inf
    likelihoods = {}

    all_tree_fits ={}
    scores = {}
    genotype_matrices = {}
    presence_matrices = {}
    for idx, tree in enumerate(tree_list):
        # genotype_matrices[idx] = get_descendants_matrix(tree)


        presence_matrices[idx] = enumerate_presence(tree, clones, clusters)
        scores[idx] =init_score(presence=presence_matrices[idx], 
                    log_like_matrix_cell_sort=log_like_matrix_cell_sort, 
                    snv_idx=snv_index_cell_sort, n_cells=n_cells, 
                    n_clones=n_clones, q_y_init=q_y_init, cell_ptr=cell_ptr, beta=beta)
    
    
    if prune:
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]
        tree_list_indices = [idx for idx, _ in top_scores]
    else:
        tree_list_indices = [i for i in range(len(tree_list))]
        #find the top ranking trees and filter tree list down to those indices
    for idx in tree_list_indices:

        """
        # Main function to find cell assignments
        raw_probability, Cell_assignment_df = (
            run(genotype_matrix, read_counts, alpha, verbose)
            find_cell_assignments(read_counts, genotype_matrix, alpha)
        )
        """

        if verbose:
            print(f"Starting tree {idx}...")
        
        # genotype_matrix = genotype_matrices[idx]
  

        presence = presence_matrices[idx]

        #run doesn't know the labels, everything is in index space
        expected_log_like, q_z, q_y = run(presence, 
                log_like_matrix_cell_sort,
                log_like_matrix_snv_sort,
                cell_idx = cell_index_snv_sort,
                snv_idx = snv_index_cell_sort,
                n_cells = n_cells,
                n_snvs = n_snvs,
                n_clones = n_clones, 
                q_y_init = q_y_init,
                cell_ptr = cell_ptr,
                snv_ptr = snv_ptr,
                max_iter = max_iter,
                tolerance = tolerance,
                beta = beta,
                verbose = verbose)

        tfit = TreeFit(tree_list[idx], idx, expected_log_like, q_z, q_y, cell_to_idx, snv_to_idx, clones)
        all_tree_fits[idx] = tfit
        if verbose:
            print(f"Tree {idx} fit wtih likelihood: {expected_log_like}")
        likelihoods[idx] = expected_log_like
        if expected_log_like > best_likelihood:
            best_fit = tfit
            best_likelihood = expected_log_like
 
    return likelihoods, best_fit, all_tree_fits
    
        

 



def main():
    args = parse_arguments()
    read_counts = pd.read_csv(args.read_counts)

    read_trees = read_tree_edges_conipher

    if args.sapling:
        read_trees = read_tree_edges_sapling

    candidate_trees = read_trees(args.trees, sep=args.edge_sep)

    likelihoods, tfit, _ = rank_trees(
        candidate_trees,
        read_counts,
        alpha=args.alpha,
        beta = args.beta,
        verbose=args.verbose,
        max_iter=args.max_iter,
        prior=args.prior
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


    print(likelihoods_df)
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
    
    if args.tree:
        tfit.save_tree(args.tree)
    
