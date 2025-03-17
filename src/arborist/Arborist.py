import sys
import os
import argparse
import pandas as pd
import numpy as np
from scipy.stats import binom
from scipy.special import logsumexp
from collections import defaultdict
from scipy.stats import entropy


def parse_arguments():
    parser = argparse.ArgumentParser(description="Tree ranking script")
    parser.add_argument(
        "--read_counts", required=True, help="Path to read counts CSV file"
    )
    parser.add_argument("--conipher", help="Path to Conipher tree file")
    parser.add_argument("--sapling", help="Path to Sapling tree file")
    parser.add_argument("--sapling_backbone", help="Path to Sapling backbone tree file")
    parser.add_argument("--output", required=True, help="Directory for output files")
    return parser.parse_args()


def read_tree_edges_conipher(file_path):
    trees = []
    current_tree = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):  # Skip empty lines and comments
                if current_tree:  # Store previous tree before starting a new one
                    trees.append(current_tree)
                    current_tree = []
                continue

            # Convert space-separated numbers to tuple (parent, child)
            parts = line.split()
            if len(parts) == 2:
                current_tree.append((int(parts[0]), int(parts[1])))

    if current_tree:  # Append last tree if exists
        trees.append(current_tree)

    return trees


def read_tree_edges_sapling(file_path, header_prefix="backbone tree"):
    trees = []
    current_tree = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):  # Skip empty lines and comments
                continue

            if line.startswith(header_prefix):  # New tree starts here
                if current_tree:  # Store previous tree before starting a new one
                    trees.append(current_tree)
                    current_tree = []
                continue

            # Convert space-separated numbers to tuple (parent, child)
            parts = line.split()
            if len(parts) == 2:
                current_tree.append((int(parts[0]), int(parts[1])))

    if current_tree:  # Append last tree if exists
        trees.append(current_tree)

    return trees


def process_read_counts_and_calculate_probabilities(
    read_counts, tree, error_rate=0.001
):
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


def rank_trees(tree_set, read_counts, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    tree_probabilities = []
    combined_outputs = []

    for idx, tree in enumerate(tree_set):
        raw_probability, Cell_assignment_df = (
            process_read_counts_and_calculate_probabilities(read_counts, tree)
        )

        # Assign each cell to the most probable clone
        assigned_clones = Cell_assignment_df.select_dtypes(include=[np.number]).idxmax(
            axis=1
        )
        Cell_assignment_df["Assigned Clone"] = assigned_clones
        Cell_assignment_df["Tree"] = f"Tree_{idx}"  # Assign tree ID for tracking
        Cell_assignment_df["Raw Probability"] = raw_probability
        combined_outputs.append(Cell_assignment_df.reset_index())

        tree_probabilities.append(
            {"Tree": f"Tree_{idx}", "Raw Probability": raw_probability}
        )

    # Create a DataFrame of tree probabilities
    df = pd.DataFrame(tree_probabilities)

    # Normalize probabilities
    logsum = logsumexp(df["Raw Probability"])
    df["Normalized Probability (%)"] = np.exp(df["Raw Probability"] - logsum) * 100

    # Rank trees by raw probability
    df.sort_values(by=["Raw Probability"], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Select top 20 ranked trees
    ranked_trees = df.iloc[:20]

    # Filter Cell_assignment_df to include only top 20 ranked trees
    filtered_assignments = pd.concat(combined_outputs, ignore_index=True)
    filtered_assignments = filtered_assignments[
        filtered_assignments["Tree"].isin(ranked_trees["Tree"])
    ]

    Entropy = filtered_assignments.iloc[:, 1:]  # Keeps all columns except the first one
    Entropy = Entropy.drop(columns=["Raw Probability", "Assigned Clone", "Tree"])
    Entropy = np.exp(Entropy - logsumexp(Entropy, axis=1, keepdims=True))
    Entropy["Entropy"] = Entropy.apply(lambda row: entropy(row, base=np.e), axis=1)
    Entropy["Cell"] = filtered_assignments["index"]
    Entropy["Tree"] = filtered_assignments["Tree"]
    Entropy["clone"] = filtered_assignments["Assigned Clone"]

    # Save results
    ranked_trees.to_csv(os.path.join(output_dir, "ranked_trees.csv"), index=False)
    filtered_assignments.to_csv(
        os.path.join(output_dir, "top20_trees_cell_assignment.csv"), index=False
    )
    Entropy.to_csv(os.path.join(output_dir, "Entropy.csv"), index=False)

    return ranked_trees, filtered_assignments, Entropy


def main():
    args = parse_arguments()
    read_counts = pd.read_csv(args.read_counts)

    if args.conipher:
        rank_trees(
            read_tree_edges_conipher(args.conipher),
            read_counts,
            os.path.join(args.output, "conipher_trees"),
        )
    if args.sapling:
        rank_trees(
            read_tree_edges_sapling(args.sapling),
            read_counts,
            os.path.join(args.output, "sapling_trees"),
        )
    if args.sapling_backbone:
        rank_trees(
            read_tree_edges_sapling(args.sapling_backbone),
            read_counts,
            os.path.join(args.output, "sapling_backbone_trees"),
        )


if __name__ == "__main__":
    main()
