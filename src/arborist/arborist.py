import argparse
import pandas as pd
import numpy as np
import numba
from numba import njit, prange
from scipy.stats import binom
from collections import defaultdict
from .utils import read_trees, visualize_tree
from .treefit import TreeFit
import networkx as nx

numba.set_num_threads(10)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arborist: a method to rank SNV clonal trees using scDNA-seq data.")
    parser.add_argument(
        "-R",
        "--read_counts",
        required=True,
        help="Path to read counts CSV file with columns 'snv', 'cell', 'cluster', 'total', 'alt'",
    )
    parser.add_argument(
        "-Y",
        "--snv-clusters",
        required=True,
        help="Path to SNV clusters CSV file with columns 'snv', 'cluster'"
    )
    parser.add_argument("-T", "--trees", required=True, help="Path to file containing all candidate trees to be ranked.")
    parser.add_argument(
        "--alpha",
        required=False,
        type=float,
        default=0.001,
        help="Per base sequencing error",
    )
    # parser.add_argument(
    #     "--normal",
    #     required=False,
    #     type=int,
    #     default=0,
    #     help="node id of the normal clone",
    # ),
    parser.add_argument(
        "--add-normal",
        required=False,
        action="store_true",
        help="add a normal clone if input trees do not already contain them",
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
        help="Path to where tree ranking output should be saved.",
    )
    parser.add_argument(
        "--cell-assign",
        required=False,
        type=str,
        help="Path to where cell assignments output should be saved.",
    )
    parser.add_argument(
        "--snv-assign",
        required=False,
        type=str,
        help="Path to where snv assignments output should be saved.",
    )
    parser.add_argument(
        "--q_y",
        required=False,
        type=str,
        help="Path to where the approximate SNV posterior should be saved",
    )
    parser.add_argument(
        "--q_z",
        required=False,
        type=str,
        help="Path to where the approximate cell posterior should be saved",
    )
    parser.add_argument(
        "--genotypes",
        required=False,
        type=str,
        help="Path to where the inferred node genotypes should be saved",
    )
    parser.add_argument(
        "-v", "--verbose", help="Print verbose output", action="store_true"
    )
    parser.add_argument(
        "-d", "--draw", required=False, type=str, help="Path to save the tree image"
    )
    parser.add_argument(
        "-t", "--tree", required=True, help="Path to save the top ranked tree file."
    )
    parser.add_argument(
         "--edge-delim", required=False, type=str, default=" ", help="edge delimiter in candidate tree file."
    )
    parser.add_argument(
        "--prior",
        required=False,
        type=float,
        default=0.7,
        help="prior (gamma) on input SNV cluster assignment",
    )
    parser.add_argument("--map-assign", action="store_true")
    parser.add_argument(
        "--pickle",
        type=str,
        help="path to where all pickled tree fits should be saved.",
    )

    return parser.parse_args()


@njit
def soft_max(arr):
    """
    Applies the softmax function row-wise to a 2D array.

    Parameters
    ----------
    arr : np.ndarray
        A 2D NumPy array.

    Returns
    -------
    np.ndarray
        The input array with softmax applied row-wise.
    """
    for j in range(arr.shape[0]):
        max_log = np.max(arr[j])
        arr[j] = np.exp((arr[j] - max_log))
        arr[j] /= np.sum(arr[j])
    return arr


@njit
def logsumexp_inline(log_terms, n):
    """
    Computes the log-sum-exp of an array for numerical stability.

    Parameters
    ----------
    log_terms : np.ndarray
        Array of log values.
    n : int
        Number of values to consider from log_terms.

    Returns
    -------
    float
        The log-sum-exp of the input values.
    """
    max_log = -np.inf
    for i in range(n):
        if log_terms[i] > max_log:
            max_log = log_terms[i]
    sum_exp = 0.0
    for i in range(n):
        sum_exp += np.exp(log_terms[i] - max_log)
    return max_log + np.log(sum_exp)


@njit(parallel=True)
def compute_q_z_sparse(
    cell_ptr, snv_idx, log_likes, log_q_y, presence, n_cells, n_clones
):
    """
    Computes the variational posterior q(z_i = r) for each cell i and clone r.

    Parameters
    ----------
    cell_ptr : np.ndarray
        Pointer array for cell indexing in sparse representation.
    snv_idx : np.ndarray
        SNV indices corresponding to each observation.
    log_likes : np.ndarray
        Log-likelihoods for each observation (interleaved present/absent).
    log_q_y : np.ndarray
        Log of the current q(y) matrix (SNV x clone-1).
    presence : np.ndarray
        Presence/absence matrix for SNV clusters and clones.
    n_cells : int
        Number of cells.
    n_clones : int
        Number of clones.

    Returns
    -------
    np.ndarray
        Posterior cell assignment probabilities q_z (n_cells x n_clones).
    """
    q_z = np.zeros((n_cells, n_clones))

    for i in prange(n_cells):
        for r in range(n_clones):
            acc = 0.0
            for k in range(cell_ptr[i], cell_ptr[i + 1]):
                log_present = log_likes[2 * k]
                log_absent = log_likes[2 * k + 1]
                j = snv_idx[k]

                log_terms = np.empty(n_clones - 1)
                for p in range(n_clones - 1):
                    log_prob = log_present * presence[p, r] + log_absent * (
                        1 - presence[p, r]
                    )
                    log_terms[p] = log_q_y[j, p] + log_prob

                acc += logsumexp_inline(log_terms, n_clones - 1)

            q_z[i, r] = acc

    return soft_max(q_z)


@njit
def compute_entropy(q):
    """
    Computes the entropy of a probability distribution matrix.

    Parameters
    ----------
    q : np.ndarray
        Probability distribution matrix (e.g., q_z or q_y).

    Returns
    -------
    float
        The total entropy of the distribution.
    """
    ent = 0.0
    for i in range(q.shape[0]):
        for j in range(q.shape[1]):
            if q[i, j] > 0:
                ent -= q[i, j] * np.log(q[i, j] + 1e-12)
    return ent


@njit
def compute_kl_divergence(q_y, q_y_init):
    """
    Computes the Kullback-Leibler (KL) divergence between two distributions.

    Parameters
    ----------
    q_y : np.ndarray
        Current variational posterior q(y).
    q_y_init : np.ndarray
        Initial/reference distribution q(y).

    Returns
    -------
    float
        KL divergence D_KL(q_y || q_y_init).
    """
    kl = 0.0
    for j in range(q_y.shape[0]):
        for p in range(q_y.shape[1]):
            kl += q_y[j, p] * (
                np.log(q_y[j, p] + 1e-12) - np.log(q_y_init[j, p] + 1e-12)
            )
    return kl


@njit
def log_array(arr):
    """
    Computes the element-wise logarithm of an array, assigning -inf where values are nonpositive.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Array of the same shape as input, with log applied to positive entries and -inf elsewhere.
    """
    log_arr = np.full(arr.shape, -np.inf, dtype=np.float64)
    for j in range(arr.shape[0]):
        for p in range(arr.shape[1]):
            if arr[j, p] > 0.0:
                log_arr[j, p] = np.log(arr[j, p])
    return log_arr


@njit(parallel=True)
def compute_q_y_sparse(
    snv_ptr, cell_idx, log_likes, log_q_z, presence, n_snvs, n_clones, log_q_y_init
):
    """
    Computes the variational posterior q(y_j = p) for each SNV j and cluster p.

    Parameters
    ----------
    snv_ptr : np.ndarray
        Pointer array for SNV indexing in sparse representation.
    cell_idx : np.ndarray
        Cell indices corresponding to each observation.
    log_likes : np.ndarray
        Log-likelihoods for each observation (interleaved present/absent).
    log_q_z : np.ndarray
        Log of the current q(z) matrix (cell x clone).
    presence : np.ndarray
        Presence/absence matrix for SNV clusters and clones.
    n_snvs : int
        Number of SNVs.
    n_clones : int
        Number of clones.
    log_q_y_init : np.ndarray
        Log of the initial/reference q(y) matrix.

    Returns
    -------
    np.ndarray
        Posterior SNV assignment probabilities q_y (n_snvs x (n_clones-1)).
    """
    q_y = np.zeros((n_snvs, n_clones - 1))

    for j in prange(n_snvs):
        for p in range(n_clones - 1):
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

            q_y[j, p] = acc + log_q_y_init[j, p]

    return soft_max(q_y)


@njit(parallel=True)
def compute_likelihood_sparse(
    cell_ptr, snv_idx, log_likes, log_q_y, log_q_z, presence, n_clones, n_cells
):
    """
    Computes the expected log-likelihood under the current variational distributions.

    Parameters
    ----------
    cell_ptr : np.ndarray
        Pointer array for cell indexing in sparse representation.
    snv_idx : np.ndarray
        SNV indices corresponding to each observation.
    log_likes : np.ndarray
        Log-likelihoods for each observation (interleaved present/absent).
    log_q_y : np.ndarray
        Log of the current q(y) matrix (SNV x cluster).
    log_q_z : np.ndarray
        Log of the current q(z) matrix (cell x clone).
    presence : np.ndarray
        Binary presence/absence matrix for SNV clusters and clones.
    n_clones : int
        Number of clones.
    n_cells : int
        Number of cells.

    Returns
    -------
    float
        The expected log-likelihood of the data under the variational distributions.
    """
    expected_log_likelihood = 0.0
    for i in prange(n_cells):
        for k in range(cell_ptr[i], cell_ptr[i + 1]):
            log_present = log_likes[2 * k]
            log_absent = log_likes[2 * k + 1]
            j = snv_idx[k]
            for r in range(n_clones):
                for p in range(n_clones - 1):
                    log_w = log_q_z[i, r] + log_q_y[j, p]
                    log_prob = log_present if presence[p, r] else log_absent
                    expected_log_likelihood += np.exp(log_w) * log_prob
    return expected_log_likelihood


def build_index_pointers(index_array, alt_idx, n, log_like_matrix):
    """
    Builds a pointer array for sparse COO representation.

    Parameters
    ----------
    index_array : np.ndarray
        Array of indices (e.g., cell or SNV indices).
    alt_idx : np.ndarray
        Second index array to be sorted accordingly.
    n : int
        Number of unique indices (cells or SNVs).
    log_like_matrix : np.ndarray
        Array of log-likelihoods (flattened).

    Returns
    -------
    tuple
        (pointer array, sorted index array, sorted alt_idx, sorted log_like_matrix)
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
    ll_mat = log_like_matrix.reshape(n_obs, 2)  # each row = [log_present, log_absent]
    ll_mat = ll_mat[sort_idx]  # apply same permutation
    log_like_matrix = ll_mat.ravel()
    return counts, index_array, alt_idx, log_like_matrix


def build_sparse_input(df, cell_to_idx, snv_to_idx):
    """
    Converts (cell, snv) log likelihood DataFrame into sparse COO-style NumPy arrays.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'cell', 'snv', 'log_present', 'log_absent'.
    cell_to_idx : dict
        Mapping from cell identifiers to indices.
    snv_to_idx : dict
        Mapping from SNV identifiers to indices.

    Returns
    -------
    tuple
        (cell_idx, snv_idx, log_matrix) where log_matrix contains interleaved log_present/log_absent.
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


def enumerate_presence(tree: list, clone_to_idx: dict, cluster_to_idx: dict) -> np.array:
    """
    Enumerates the presence of each clone in the genotype matrix.

    Parameters
    ----------
    tree : list
        List of edges representing the tree (as (parent, child) tuples).
    clones : list
        List of clone identifiers.
    clusters : list
        List of cluster identifiers (subset of clones).

    Returnstheta
    -------
    np.ndarray
        Presence matrix of shape (n_clusters, n_clones), where entry [p, r] is 1 if clone r is a descendant of cluster p.
    """
    T = nx.DiGraph(tree)

    presence = np.zeros((len(cluster_to_idx), len(clone_to_idx)), dtype=int)
    # clone_to_idx = {r: i for i, r in enumerate(clones)}

    for  cluster_p, idx_p in cluster_to_idx.items():
        desc = nx.descendants(T, cluster_p) | {cluster_p}
        for d in desc:
            presence[idx_p, clone_to_idx[d] ] = 1
    # print(presence)
    return presence


def initialize_q_y(
    snv_clusters: pd.DataFrame, cluster_to_idx: dict, snv_to_idx: dict, gamma: float
) -> np.ndarray:
    """
    Initialize the variational posterior q(y_j = p) for each SNV j and cluster p.

    Parameters
    ----------
    snv_clusters : pd.DataFrame
        Must have columns 'snv' and 'cluster', where 'cluster' is the
        initial hard assignment ψ(snv) in [clusters].
    clusters : list
        List of cluster identifiers (length k).
    snv_to_idx : dict
        Mapping from SNV identifiers to row indices 0..m-1 in the output array.
    gamma : float
        Probability mass assigned to the initial cluster assignment.

    Returns
    -------
    np.ndarray
        q_y[j, p] = gamma if ψ(j) == p else (1-gamma)/(k-1).
    """
    # Build a quick map: snv -> its hard cluster
    cluster_map = dict(zip(snv_clusters["snv"], snv_clusters["cluster"]))
    m, k = len(snv_to_idx), len(cluster_to_idx)
    q_y = np.zeros((m, k), dtype=np.float64)

    # Epsilon mass on correct cluster; uniform remainder
    other = (1 - gamma) / (k - 1) if k > 1 else 0.0

    for snv, j in snv_to_idx.items():
        
        assigned = cluster_map[snv]
        for cluster, idx in cluster_to_idx.items():
            q_y[j, idx] = gamma if assigned == cluster else other

    return q_y


@njit
def hard_assign_q(q_dist: np.ndarray):
    """
    Computes the hard assignment (MAP) for a probability distribution matrix.

    Parameters
    ----------
    q_dist : np.ndarray
        Probability distribution matrix (e.g., q_z or q_y).

    Returns
    -------
    np.ndarray
        Hard assignment matrix of the same shape as q_dist, with 1 at the MAP assignment per row and 0 elsewhere.
    """
    q_hard = np.zeros(q_dist.shape)
    assign = q_dist.argmax(axis=1)
    for i in range(q_dist.shape[0]):
        for q in range(q_dist.shape[1]):
            if assign[i] == q:
                q_hard[i, q] = 1.0
            else:
                q_hard[i, q] = 0.0
    return q_hard


@njit
def run_simple_max_likelihood(
    presence: np.ndarray,
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
):
    """
    Runs a single round of MAP assignment for cells and SNVs (no variational updates).

    Parameters
    ----------
    presence : np.ndarray
        Presence/absence matrix for SNV clusters and clones.
    log_like_matrix_cell_sort : np.ndarray
        Log-likelihoods sorted by cell.
    log_like_matrix_snv_sort : np.ndarray
        Log-likelihoods sorted by SNV.
    cell_idx : np.ndarray
        Cell indices for SNV-sorted data.
    snv_idx : np.ndarray
        SNV indices for cell-sorted data.
    n_cells : int
        Number of cells.
    n_snvs : int
        Number of SNVs.
    n_clones : int
        Number of clones.
    q_y_init : np.ndarray
        Initial q(y) matrix.
    cell_ptr : np.ndarray
        Pointer array for cell indexing.
    snv_ptr : np.ndarray
        Pointer array for SNV indexing.
    max_iter : int, optional
        Not used (for API compatibility).
    tolerance : float, optional
        Not used (for API compatibility).

    Returns
    -------
    tuple
        (likelihood, q_z, q_y_hard)
    """
    # Compute initial q_z and ELBO before any updates
    q_y_hard = hard_assign_q(q_y_init)
    log_q_y_init = log_array(q_y_hard)

    q_z = compute_q_z_sparse(
        cell_ptr,
        snv_idx,
        log_like_matrix_cell_sort,
        log_q_y_init,
        presence,
        n_cells,
        n_clones,
    )
    q_z_hard = hard_assign_q(q_z)
    log_q_z = log_array(q_z_hard)
    likelihood = compute_likelihood_sparse(
        cell_ptr,
        snv_idx,
        log_like_matrix_cell_sort,
        log_q_y_init,
        log_q_z,
        presence,
        n_clones,
        n_cells,
    )
    return likelihood, q_z, q_y_hard


@njit
def run_variational_inference(
    presence: np.ndarray,
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
):
    """
    Runs coordinate ascent variational inference for cell and SNV assignments.

    Parameters
    ----------
    presence : np.ndarray
        Presence/absence matrix for SNV clusters and clones.
    log_like_matrix_cell_sort : np.ndarray
        Log-likelihoods sorted by cell.
    log_like_matrix_snv_sort : np.ndarray
        Log-likelihoods sorted by SNV.
    cell_idx : np.ndarray
        Cell indices for SNV-sorted data.
    snv_idx : np.ndarray
        SNV indices for cell-sorted data.
    n_cells : int
        Number of cells.
    n_snvs : int
        Number of SNVs.
    n_clones : int
        Number of clones.
    q_y_init : np.ndarray
        Initial/reference q(y) matrix.
    cell_ptr : np.ndarray
        Pointer array for cell indexing.
    snv_ptr : np.ndarray
        Pointer array for SNV indexing.
    max_iter : int, optional
        Maximum number of coordinate ascent iterations.
    tolerance : float, optional
        Convergence tolerance for ELBO.

    Returns
    -------
    tuple
        (best_elbo, best_q_z, best_q_y)
    """
    # Compute initial q_z and ELBO before any updates
    log_q_y_init = log_array(q_y_init)
    log_q_y = log_q_y_init.copy()
    q_z = compute_q_z_sparse(
        cell_ptr,
        snv_idx,
        log_like_matrix_cell_sort,
        log_q_y,
        presence,
        n_cells,
        n_clones,
    )
    log_q_z = log_array(q_z)
    initial_likelihood = compute_likelihood_sparse(
        cell_ptr,
        snv_idx,
        log_like_matrix_cell_sort,
        log_q_y,
        log_q_z,
        presence,
        n_clones,
        n_cells,
    )
    best_elbo = (
        initial_likelihood
        + compute_entropy(q_z)
        - compute_kl_divergence(q_y_init, q_y_init)
    )

    # Initialize best values to initial state
    best_q_z = q_z.copy()
    best_q_y = q_y_init.copy()

    converged = False

    for it in range(max_iter):
        q_y = compute_q_y_sparse(
            snv_ptr,
            cell_idx,
            log_like_matrix_snv_sort,
            log_q_z,
            presence,
            n_snvs,
            n_clones,
            log_q_y_init,
        )
        log_q_y = log_array(q_y)

        q_z = compute_q_z_sparse(
            cell_ptr,
            snv_idx,
            log_like_matrix_cell_sort,
            log_q_y,
            presence,
            n_cells,
            n_clones,
        )
        log_q_z = log_array(q_z)

        likelihood = compute_likelihood_sparse(
            cell_ptr,
            snv_idx,
            log_like_matrix_cell_sort,
            log_q_y,
            log_q_z,
            presence,
            n_clones,
            n_cells,
        )
        kl_penalty = compute_kl_divergence(q_y, q_y_init)
        cell_entropy = compute_entropy(q_z)
        elbo = likelihood + cell_entropy - kl_penalty

        if np.abs(elbo - best_elbo) < tolerance:
            converged = True

        if elbo > best_elbo:
            best_q_z = q_z.copy()
            best_q_y = q_y.copy()
            best_elbo = elbo

        if converged:
            break

    return best_elbo, best_q_z, best_q_y


def precompute_log_likelihoods(
    read_counts: pd.DataFrame, error_rate=0.001
) -> pd.DataFrame:
    """
    Preprocesses the read counts by precomputing the log probabilities.

    Parameters
    ----------
    read_counts : pd.DataFrame
        DataFrame with columns 'cell', 'snv', 'alt', 'total', etc.
    error_rate : float, optional
        Per-base sequencing error rate (default is 0.001).

    Returns
    -------
    tuple
        (read_counts_with_logs, cell_to_idx, snv_to_idx)
    """
    read_counts["log_absent"] = binom.logpmf(
        read_counts["alt"], read_counts["total"], error_rate / 3
    )
    read_counts["log_present"] = binom.logpmf(
        read_counts["alt"],
        read_counts["total"],
        0.5 - error_rate + 0.5 * error_rate / 3,
    )

    cells = read_counts["cell"].unique()
    snvs = read_counts["snv"].unique()

    cell_to_idx = {cell: idx for idx, cell in enumerate(cells)}
    snv_to_idx = {snv: idx for idx, snv in enumerate(snvs)}


    return read_counts, cell_to_idx, snv_to_idx


def tree_to_clone_set(tree: list) -> list:
    """
    Extracts the set of unique clones from a tree edge list.

    Parameters
    ----------
    tree : list
        List of edges (parent, child) tuples.

    Returns
    -------
    set
        Set of unique clone identifiers in the tree.
    """
    clones = set()
    for u, v in tree:
        clones.add(u)
        clones.add(v)
    return clones


def rank_trees(
    tree_list: list,
    read_counts: pd.DataFrame,
    snv_clusters: pd.DataFrame,
    alpha: float = 0.001,
    max_iter: int = 10,
    tolerance: float = 1,
    gamma=0.7,
    update_snvs=True,
    verbose: bool = False
) -> tuple:
    """
    Rank SNV phylogenetic trees based on evidence lower bound given scDNA-seq read count data.
    This function processes a list of clonal trees and their associated read counts to calculate
    the likelihood of each tree.
    Parameters
    ----------
    tree_list : list
        A list of phylogenetic trees to be ranked.
    read_counts : pandas.DataFrame
        A DataFrame containing columns ["snv", "cell", "alt", "total", "cluster"]
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
    >>> ranked_trees, filtered_assignments, all_fits = rank_trees(tree_list, read_counts, alpha=0.001)

    """

    tree = tree_list[0]
    
    #assume root is normal
    temp_tree = nx.DiGraph(tree)
    normal = [n for n in temp_tree if temp_tree.in_degree(n)==0][0]
    clone_set = tree_to_clone_set(tree)
    for tree in tree_list:
        if tree_to_clone_set(tree) != clone_set:
            raise ValueError("All trees must have the same set of clones.")

    clones = list(clone_set)

    clusters = [c for c in clones if c != normal]
  
    # Filter read_counts to only include cells and SNVs present in the tree
    # read_counts = read_counts[read_counts["cluster"].isin(clones)]
    clones.sort()
    clusters.sort()
    clone_to_idx = {c: i for i,c in enumerate(clones)}
    cluster_to_idx = {c: i for i,c in enumerate(clusters)}

    alt_sum = read_counts.groupby("snv")["alt"].sum()
    valid_snvs = alt_sum[alt_sum > 0].index

    print(f"Number of valid SNVs:  {len(valid_snvs)}")
    read_counts = read_counts[read_counts["snv"].isin(valid_snvs)]

    # appends columns log_absent and log_present to read_counts
    read_counts, cell_to_idx, snv_to_idx = precompute_log_likelihoods(
        read_counts, alpha
    )
    q_y_init = initialize_q_y(snv_clusters, cluster_to_idx, snv_to_idx, gamma)
    cell_idx, snv_idx, log_like_matrix = build_sparse_input(
        read_counts, cell_to_idx, snv_to_idx
    )

    n_cells = len(cell_to_idx)
    n_snvs = len(snv_to_idx)
    n_clones = len(clones)
    # cell_ptr, cell_sort_idx = build_index_pointers(cell_idx, snv_idx, n_cells)
    cell_ptr, cell_idx_sort, snv_index_cell_sort, log_like_matrix_cell_sort = (
        build_index_pointers(
            cell_idx, snv_idx, n_cells, log_like_matrix=log_like_matrix
        )
    )
    snv_ptr, snv_idx_sort, cell_index_snv_sort, log_like_matrix_snv_sort = (
        build_index_pointers(snv_idx, cell_idx, n_snvs, log_like_matrix=log_like_matrix)
    )



    best_likelihood = -np.inf
    likelihoods = {}

    all_tree_fits = {}

    if update_snvs:
        if verbose:
            print("Running Arborist in full variational inference mode...")
        run = run_variational_inference
    else:
        print("Running Arborist in cell MAP assignment mode....")
        run = run_simple_max_likelihood

     
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



        presence = enumerate_presence(tree, clone_to_idx, cluster_to_idx)
      

        expected_log_like, q_z, q_y = run(
            presence,
            log_like_matrix_cell_sort,
            log_like_matrix_snv_sort,
            cell_idx=cell_index_snv_sort,
            snv_idx=snv_index_cell_sort,
            n_cells=n_cells,
            n_snvs=n_snvs,
            n_clones=n_clones,
            q_y_init=q_y_init,
            cell_ptr=cell_ptr,
            snv_ptr=snv_ptr,
            max_iter=max_iter,
            tolerance=tolerance,
        )

        tfit = TreeFit(
            tree_list[idx],
            idx,
            expected_log_like,
            q_z,
            q_y,
            cell_to_idx,
            snv_to_idx,
            clone_to_idx,
            cluster_to_idx
        )
        all_tree_fits[idx] = tfit
        if verbose:
            print(f"Tree {idx} fit wtih ELBO: {expected_log_like}")
        likelihoods[idx] = expected_log_like
        if expected_log_like > best_likelihood:
            best_fit = tfit
            best_likelihood = expected_log_like

    return likelihoods, best_fit, all_tree_fits


def main():
    args = parse_arguments()
    read_counts = pd.read_csv(args.read_counts)

    #SNV cluster input should include no column names
    snv_clusters = pd.read_csv(args.snv_clusters, header=None, names=["snv", "cluster"])


  
    candidate_trees = read_trees(args.trees, sep=args.edge_delim)
    for t in candidate_trees:
            print(t)

    if args.add_normal:
        print("appending normal clone to trees...")
        cand_tree_with_root= []
        for edge_list in candidate_trees:
            tree = nx.DiGraph(edge_list)
            root = [n for n in tree if len(list(tree.predecessors(n)))==0][0]
            normal = min(list(tree.nodes)) - 1
            tree.add_edge(normal, root)
            cand_tree_with_root.append(list(tree.edges))
        candidate_trees=cand_tree_with_root
        for t in candidate_trees:
            print(t)

 



    elbos, tfit, all_fits = rank_trees(
        candidate_trees,
        read_counts,
        snv_clusters,
        alpha=args.alpha,
        verbose=args.verbose,
        max_iter=args.max_iter,
        gamma=args.prior,
        update_snvs=not args.map_assign,
    )

    if args.draw:

        visualize_tree(
            tfit.tree,
            output_file=args.draw,
        )

    cell_assign = tfit.map_assign_z()
    snv_assign = tfit.map_assign_y()

    elbo_df = pd.DataFrame.from_dict(elbos, orient="index", columns=["elbo"])
    elbo_df = elbo_df.reset_index().rename(columns={"index": "tree_idx"})
    elbo_df = elbo_df.sort_values(by="elbo", ascending=False)

    # # Save results
    if args.ranking:
        elbo_df.to_csv(args.ranking, index=False)

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

    if args.genotypes:
        tfit.save_genotypes(args.genotypes)

    if args.pickle:
        pd.to_pickle(all_fits, args.pickle)
