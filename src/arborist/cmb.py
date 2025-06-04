# Created by: L.L. Weber
# Created on: 2024-05-06 09:20:11
# Last modified on: 2024-05-06 09:20:11


import pandas as pd
import itertools
import argparse
from collections import defaultdict
from .utils import (
    read_tree_edges_conipher,
    read_tree_edges_sapling,
    get_descendants,
    edge_list_to_adj_list,
    get_nodes,
)


def compute_cmb(df, cells, snvs):
    df = df[df["cell"].isin(cells)]
    df = df[df["snv"].isin(snvs)]
    df_cmb = (
        df.groupby("cell")
        .apply(lambda x: (x["alt"] > 0).mean())
        .reset_index(name="cmb")
    )
    return df_cmb


def clade_cmb(n, cell_mapping, snvs, T, df, min_cells=10):

    clade_nodes = get_descendants(T, n) | {n}

    outside_nodes = get_nodes(T) - clade_nodes

    within_clade_cells = list(
        itertools.chain.from_iterable([cell_mapping[u] for u in clade_nodes])
    )
    outside_clade_cells = list(
        itertools.chain.from_iterable([cell_mapping[u] for u in outside_nodes])
    )

    outside_df = None
    within_df = None
    # returns a vector of length outside_clade_cells
    if len(outside_clade_cells) >= min_cells:
        outside_df = compute_cmb(df, outside_clade_cells, snvs)

        outside_df["within_clade"] = 0

    if len(within_clade_cells) >= min_cells:
        within_df = compute_cmb(df, within_clade_cells, snvs)
        within_df["within_clade"] = 1

    if outside_df is not None and within_df is not None:
        df = pd.concat([outside_df, within_df])
    elif outside_df is not None:
        df = outside_df
    elif within_df is not None:
        df = within_df
    else:
        df = pd.DataFrame(columns=["cell", "cmb", "within_clade"])
    df["clade_snvs"] = len(snvs)
    df["clade"] = n

    return df


def cmb(ca_df, trees, clade_snvs, read_counts, min_cells=10):

    tree_index = ca_df["tree"].unique()
    assert len(tree_index) == 1
    tree = trees[tree_index[0]]
    tree_dict = edge_list_to_adj_list(tree)

    phi = dict(zip(ca_df["cell"], ca_df["clone"]))
    cell_mapping = defaultdict(list)
    for cell, node in phi.items():
        cell_mapping[node].append(cell)

    cmb_list = []
    for n in get_nodes(tree_dict):
        snvs = clade_snvs[n]
        if len(snvs) > 0:
            cmb_list.append(
                clade_cmb(
                    n, cell_mapping, snvs, tree_dict, read_counts, min_cells=min_cells
                )
            )

    cmb_df = pd.concat(cmb_list)
    cmb_df["tree"] = tree_index[0]

    return cmb_df


def main():
    args = parse_args()
    dat = pd.read_csv(args.read_counts)

    phi_df = pd.read_csv(args.cell_assign)

    psi = dat[["snv", "cluster"]].drop_duplicates()
    psi = dict(zip(psi["snv"], psi["cluster"]))
    clade_snvs = defaultdict(list)
    for j, cluster in psi.items():
        clade_snvs[cluster].append(j)

    read_trees = read_tree_edges_conipher
    if args.sapling:
        read_trees = read_tree_edges_sapling

    trees = read_trees(args.trees)

    cmb_df = phi_df.groupby("tree").apply(
        lambda x: cmb(x, trees, clade_snvs, dat, min_cells=args.min_cells)
    )

    if args.out:
        cmb_df.to_csv(args.out, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-R",
        "--read_counts",
        required=True,
        help="Path to read counts CSV file with columns 'cell', 'cluster', 'total', 'alt'",
    )
    parser.add_argument("-T", "--trees", required=True, help="Path to tree file")
    parser.add_argument(
        "--cell-assign",
        required=False,
        type=str,
        help="Path to where cell assignments, with columns 'cell', 'tree', 'clone'",
    )
    parser.add_argument(
        "--sapling",
        action="store_true",
        help="Use sapling format for tree edges, otherwise conipher format is assumed.",
    )
    parser.add_argument(
        "--min-cells",
        required=False,
        type=int,
        default=10,
        help="minimum number of cells to compute the scores for a clade",
    )
    parser.add_argument(
        "-o",
        "--out",
        required=False,
        type=str,
        help="filename where cmb values will be saved",
    )
    parser.add_argument(
        "--conipher", action="store_true", help="if the tree is in conipher format"
    )

    # args = parser.parse_args([
    #     "--data", "GEM2.2/read_counts.csv",
    #     "--tree", "GEM2.2/conipher.txt",
    #     # "--phi", "top20_trees_cell_assignment.csv",
    #     "--phi", "cell_assignment_conipher.csv",
    #     "--min-cells", "20",
    #     "--out", "cmb.csv",
    #     "--tree-path",  "GEM2.2/conipher_trees",
    #     "--conipher",
    # ])

    return parser.parse_args()
