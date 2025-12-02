import argparse
import pandas as pd
import numpy as np
import numba
from numba import njit, prange
from scipy.stats import binom
from .utils import read_trees, visualize_tree
from .treefit import TreeFit
import networkx as nx
import warnings
import logging


logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


# warnings.filterwarnings("ignore")



def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Arborist: a method to rank SNV clonal trees using scDNA-seq data."
    )
    parser.add_argument(
        "-R",
        "--read_counts",
        required=True,
        help="Path to read counts CSV file with columns 'snv', 'cell',  'total', 'alt'",
    )
    parser.add_argument(
        "-Y",
        "--snv-clusters",
        required=True,
        help="Path to SNV clusters CSV file with unlabeled columns 'snv', 'cluster'. The order of columns matters",
    )
    parser.add_argument(
        "-T",
        "--trees",
        required=True,
        help="Path to file containing all candidate trees to be ranked.",
    )
    parser.add_argument(
        "--edge-delim",
        required=False,
        type=str,
        default=" ",
        help="edge delimiter in candidate tree file.",
    )
    parser.add_argument(
        "--add-normal",
        required=False,
        action="store_true",
        help="flag to add a normal clone if input trees do not already contain them",
    )
    parser.add_argument(
        "--alpha",
        required=False,
        type=float,
        default=0.001,
        help="Per base sequencing error",
    )
    parser.add_argument(
        "--max-iter",
        required=False,
        type=int,
        default=25,
        help="max number of iterations",
    )
    parser.add_argument(
        "--prior",
        required=False,
        type=float,
        default=0.7,
        help="prior (gamma) on input SNV cluster assignment",
    )
    parser.add_argument(
        "--pickle",
        type=str,
        help="path to where all pickled tree fits should be saved.",
    )
    parser.add_argument(
        "-d",
        "--draw",
        required=False,
        type=str,
        help="Path to where the tree visualization should be saved",
    )
    parser.add_argument(
        "-t",
        "--tree",
        required=False,
        help="Path to save the top ranked tree as a txt file.",
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
        help="Path to where the MAP cell-to-clone labels should be saved",
    )
    parser.add_argument(
        "--snv-assign",
        required=False,
        type=str,
        help="Path to where the MAP SNV-to-cluster labels should be saved.",
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
        "-j",
        "--threads",
        required=False,
        default= 10,
        type=int,
        help="Number of threads to use",
    )
    parser.add_argument(
        "-v", "--verbose", help="Print verbose output", action="store_true"
    )

    return parser.parse_args()


# parser.add_argument(
#     "--genotypes",
#     required=False,
#     type=str,
#     help="Path to where the inferred node genotypes should be saved",
# )
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
        KL divergence KL(q_y || q_y_init).
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


def enumerate_presence(
    tree: list, clone_to_idx: dict, cluster_to_idx: dict
) -> np.array:
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

    for cluster_p, idx_p in cluster_to_idx.items():
        desc = nx.descendants(T, cluster_p) | {cluster_p}
        for d in desc:
            presence[idx_p, clone_to_idx[d]] = 1

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
    clusters_to_idx : dict
        Mapping of clusters to their index
    snv_to_idx : dict
        Mapping from SNV identifiers to row indices 0..m-1 in the output array.
    gamma : float
        Probability mass assigned to the initial cluster assignment.

    Returns
    -------
    np.ndarray
        q_y[j, p] = gamma if psi(j) == p else (1-gamma)/(k-1).
    """
    # Build a quick map: snv -> its hard cluster
    cluster_map = dict(zip(snv_clusters["snv"], snv_clusters["cluster"]))
    m, k = len(snv_to_idx), len(cluster_to_idx)
    q_y = np.zeros((m, k), dtype=np.float64)

    # Epsilon mass on correct cluster; uniform remainder
    other = (1 - gamma) / (k - 1) if k > 1 else 0.0

    uniform = 1 / k

    for snv, j in snv_to_idx.items():
        if snv not in cluster_map:
            print(f"SNV {snv} not initialized with cluster assignment.")
            assigned = None
        else:
            assigned = cluster_map[snv]

        for cluster, idx in cluster_to_idx.items():
            if not assigned or assigned not in cluster_to_idx:
                q_y[j, idx] = uniform
            else:
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
        Presence/absence binary matrix for SNV clusters and clones describing the clone tree.
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
    Preprocesses the read counts by precomputing the log probabilities for both presence and absence.

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

    read_counts.loc[:, "log_present"] = binom.logpmf(
        read_counts["alt"],
        read_counts["total"],
        0.5 - error_rate + 0.5 * error_rate / 3,
    )

    read_counts.loc[:,"log_absent"] = binom.logpmf(
        read_counts["alt"], read_counts["total"], error_rate / 3
    )




    cells = read_counts["cell"].unique()
    snvs = read_counts["snv"].unique()

    cell_to_idx = {cell: idx for idx, cell in enumerate(cells)}
    snv_to_idx = {snv: idx for idx, snv in enumerate(snvs)}

    return read_counts, cell_to_idx, snv_to_idx


def add_normal_clone(candidate_trees: list):
    """
     Adds a normal clone at the root of each tree in the candidate set.

     Parameters
     ----------
     cadidate_trees : a list of lists of edge tuples (u,v)
         list of edge lists for each candidate tree in the set

     Returns
     -------
    list
        a list of lists of edge tuples (u,v) with a new edge appended from the normal cell to the clonal cluster
    """
    cand_tree_with_root = []
    for edge_list in candidate_trees:
        tree = nx.DiGraph(edge_list)
        root = [n for n in tree if len(list(tree.predecessors(n))) == 0][0]
        normal = min(list(tree.nodes)) - 1
        tree.add_edge(normal, root)
        cand_tree_with_root.append(list(tree.edges))
    candidate_trees = cand_tree_with_root
    return candidate_trees


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


def arborist(
    tree_list: list,
    read_counts: pd.DataFrame,
    snv_clusters: pd.DataFrame,
    alpha: float = 0.001,
    max_iter: int = 10,
    tolerance: float = 1,
    gamma=0.7,
    add_normal=False,
    threads = 10,
    verbose: bool = False,
) -> tuple:
    """
    Rank candidate SNV phylogenetic trees using scDNA-seq read counts.

    This function evaluates a list of clonal trees using a variational inference
    scheme and returns an evidence lower bound (ELBO)-based score for each tree.
    An initial SNV-to-cluster prior is set given an initial SNV clustering, and both cell-to-clone and
    SNV-to-cluster assignments are optimized under the model.

    Parameters
    ----------
    tree_list : list of list of tuple
        Candidate phylogenetic trees to be ranked. Each tree is represented as
        a list of directed edges ``(parent, child)``. All trees must contain
        the same set of clone identifiers.
    read_counts : pandas.DataFrame
        Data frame with columns ``["snv", "cell", "alt", "total"]`` in any order, 
        containing per-cell read counts for each SNV.
    snv_clusters : pandas.DataFrame
        Data frame with columns ``["snv", "cluster"]`` in any order, giving an
        initial hard assignment of SNVs to clusters.
    alpha : float, optional
        Per-base sequencing error rate used to compute log-likelihoods for
        presence/absence of an SNV (default is ``0.001``).
    max_iter : int, optional
        Maximum number of coordinate-ascent iterations in the variational
        inference procedure (default is ``10``).
    tolerance : float, optional
        Convergence threshold on the change in ELBO between iterations
        (default is ``1.0``).
    gamma : float, optional
        Prior probability mass placed on the initial SNV cluster assignment
        in ``q_y`` (default is ``0.7``). The remaining mass is spread
        uniformly over alternative clusters.
    add_normal : bool, optional
        If ``True``, prepend a normal clone as a new root node to each tree in
        ``tree_list`` (default is ``False``).
    threads : int, optional
        Number of threads to use for numba-parallelized computations
        (default is ``10``).
    verbose : bool, optional
        If ``True``, enable informative logging messages during fitting
        (default is ``False``).

    Returns
    -------
    likelihoods : dict[int, float]
        Mapping from tree index to its ELBO (expected log joint) under the
        variational posterior.
    best_fit : TreeFit
        ``TreeFit`` object for the top-ranked tree, containing the tree,
        ELBO, posterior cell-to-clone distribution ``q_z``, posterior
        SNV-to-cluster distribution ``q_y``, and associated index mappings.
    all_tree_fits : dict[int, TreeFit]
        Dictionary mapping each tree index to its corresponding ``TreeFit``
        object.

    Raises
    ------
    ValueError
        If the candidate trees in ``tree_list`` do not all share the same
        set of clone identifiers.

    Notes
    -----
    Only SNVs with at least one variant read across all cells are retained
    for inference. SNVs with zero total variant counts are dropped from
    ``read_counts`` before fitting.

    Examples
    --------
    >>> likelihoods, best_fit, all_fits = arborist(
    ...     tree_list,
    ...     read_counts,
    ...     snv_clusters,
    ...     alpha=0.001,
    ...     max_iter=25,
    ...     gamma=0.7,
    ... )
    """


    if verbose:
        logger = logging.getLogger()
    
    numba.set_num_threads(threads)

    if add_normal:
        if verbose:
            logger.info(f"Appending normal clone to candidate trees...")
        tree_list = add_normal_clone(tree_list)

    tree = tree_list[0]

    # assume root is normal
    temp_tree = nx.DiGraph(tree)
    normal = [n for n in temp_tree if temp_tree.in_degree(n) == 0][0]
    clone_set = tree_to_clone_set(tree)
    for tree in tree_list:
        if tree_to_clone_set(tree) != clone_set:
            raise ValueError("All trees must have the same set of clones.")

    clones = list(clone_set)

    clusters = [c for c in clones if c != normal]

    # Filter read_counts to only include cells and SNVs present in the tree

    clones.sort()
    clusters.sort()
    clone_to_idx = {c: i for i, c in enumerate(clones)}
    cluster_to_idx = {c: i for i, c in enumerate(clusters)}


    if verbose:
        logger.info(f"Removing SNVs with 0 variant reads across all cells...")

    alt_sum = read_counts.groupby("snv")["alt"].sum()
    valid_snvs = alt_sum[alt_sum > 0].index


    # valid SNVs are SNVs that have at least one variant read across all cells
    # otherwise we have no signal to place them in the tree

    read_counts = read_counts[read_counts["snv"].isin(valid_snvs)].copy()

    # appends columns log_absent and log_present to read_counts
    if verbose:
        logger.info(f"Caching log-likelihoods for presence/absence...")
    read_counts, cell_to_idx, snv_to_idx = precompute_log_likelihoods(
        read_counts, alpha
    )

    if verbose:
        logger.info(f"Initializing the SNV cluster assignment prior...")
    q_y_init = initialize_q_y(snv_clusters, cluster_to_idx, snv_to_idx, gamma)
    cell_idx, snv_idx, log_like_matrix = build_sparse_input(
        read_counts, cell_to_idx, snv_to_idx
    )

    n_cells = len(cell_to_idx)
    n_snvs = len(snv_to_idx)
    n_clones = len(clones)


    cell_ptr, _, snv_index_cell_sort, log_like_matrix_cell_sort = (
        build_index_pointers(
            cell_idx, snv_idx, n_cells, log_like_matrix=log_like_matrix
        )
    )
    snv_ptr, _ , cell_index_snv_sort, log_like_matrix_snv_sort = (
        build_index_pointers(snv_idx, cell_idx, n_snvs, log_like_matrix=log_like_matrix)
    )

    best_likelihood = -np.inf
    likelihoods = {}

    all_tree_fits = {}

    run = run_variational_inference

    if verbose:
        logger.info(f"Starting Arborist...")
        logger.info(f"Number of candidate trees: {len(tree)}")
        logger.info(f"Number of clones: {n_clones}")
        logger.info(f"Number of SNV clusters: {len(cluster_to_idx)}")
        logger.info(f"Number of cells: {n_cells}")
        logger.info(f"Number of SNVs: {n_snvs}")
    # else:
    #     print("Running Arborist in cell MAP assignment mode....")
    #     run = run_simple_max_likelihood

    for idx, tree in enumerate(tree_list):

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
            cluster_to_idx,
        )
        all_tree_fits[idx] = tfit
        if verbose:
            logger.info(f"Tree {idx} fit wtih ELBO: {expected_log_like:.2f}")
        likelihoods[idx] = expected_log_like
        if expected_log_like > best_likelihood:
            best_fit = tfit
            best_likelihood = expected_log_like

    if verbose:
        logger.info(f"Done fitting all {len(tree_list)} candidate trees!")
    return likelihoods, best_fit, all_tree_fits


def main():
    args = parse_arguments()
    read_counts = pd.read_csv(args.read_counts)

    # SNV cluster input should include no column names
    snv_clusters = pd.read_csv(args.snv_clusters, header=None, names=["snv", "cluster"])

    candidate_trees = read_trees(args.trees, sep=args.edge_delim)

    elbos, tfit, all_fits = arborist(
        candidate_trees,
        read_counts,
        snv_clusters,
        alpha=args.alpha,
        verbose=args.verbose,
        max_iter=args.max_iter,
        gamma=args.prior,
        add_normal=args.add_normal,
        threads = args.threads
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

    # if args.genotypes:
    #     tfit.save_genotypes(args.genotypes)

    if args.pickle:
        pd.to_pickle(all_fits, args.pickle)

    print("\n---------------------Arborist complete!---------------------\n")
    print(tfit)
