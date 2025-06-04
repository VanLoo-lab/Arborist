from dataclasses import dataclass
import pandas as pd
import numpy as np
import pygraphviz as pgv
from collections import defaultdict


@dataclass
class TreeFit:
    """
    Solution class to hold the solution to a tree
    """

    tree: list
    tree_idx: int
    elbo: float
    q_z: np.ndarray
    q_y: np.ndarray
    cell_to_idx: dict
    snv_to_idx: dict
    clones: list

    def __str__(self):
        return f"{self.tree_idx}: {self.elbo}"

    def convert_q_to_dataframe(self, q, my_dict):
        """
        converts q to a dataframe
        """
        df = pd.DataFrame(q)
        df.columns = [f"clone_{self.clones[i]}" for i in range(q.shape[1])]
        df.index.name = "id"
        df.reset_index(inplace=True)
        label_dict = {val: key for key, val in my_dict.items()}
        df["id"] = df["id"].map(label_dict)

        return df

    def q_z_df(self):
        df = self.convert_q_to_dataframe(self.q_z, self.cell_to_idx)

        return df

    def q_y_df(self):
        df = self.convert_q_to_dataframe(self.q_y, self.snv_to_idx)

        return df

    def __repr__(self):
        return f"TreeFit(tree={self.tree}, tree_idx={self.tree_idx}, expected_log_likelihood={self.elbo})"

    @staticmethod
    def map_assign(q, mydict):
        """
        assigns cell to the maximum a posteriori (MAP) clone/cluster
        """
        q_assign = q.argmax(axis=1)
        q_df = pd.DataFrame(q_assign, columns=["assignment"])
        q_df.index.name = "id"
        q_df.reset_index(inplace=True)
        label_dict = {val: key for key, val in mydict.items()}
        q_df["id"] = q_df["id"].map(label_dict)

        return q_df

    def map_assign_z(self):
        """
        assigns cell to the maximum a posteriori (MAP) clone
        """
        return self.map_assign(self.q_z, self.cell_to_idx)

    def map_assign_y(self):
        """
        assigns snvs to the maximum a posteriori (MAP) cluster
        """
        return self.map_assign(self.q_y, self.snv_to_idx)

    def save_tree(self, fname):
        pass
        with open(fname, "w+") as file:
            file.write("# tree\n")
            for u, v in self.tree:
                file.write(f"{u},{v}\n")

    def visualize_tree(self, include_scores=False, output_file=None):
        """
        Visualizes a tree using Graphviz.

        Parameters
        ----------
        tree : list of tuple of int
            The tree to visualize, represented as a list of tuples where each tuple
            contains two integers representing a parent
            and
            child relationship.
        cell_assignment : dict of int to int, optional
            A dictionary mapping each node to a cell number. If provided, the nodes
            will be labeled by the number of assigned cells.
        output_file : str, optional
            The path to save the visualization. If not provided, the visualization
            will be displayed on the screen.        "
        """
        labels = defaultdict(str)

        if include_scores:
            cs = self.clade_score()

            for parent, child in self.tree:
                labels[parent] = f"{parent}\nscore: {cs[parent]:.3f}"
                labels[child] = f"{child}\nscore: {cs[child]:.3f}"
        else:
            for parent, child in self.tree:
                labels[parent] = f"{parent}"
                labels[child] = f"{child}"

        # if cell_assignment:
        #     cell_mapping = defaultdict(list)
        #     for node, cell in cell_assignment.items():
        #         cell_mapping[cell].append(node)

        #     for node, cells in cell_mapping.items():
        #         labels[node] = f"{node}\n({len(cells)} cells)"

        graph = pgv.AGraph(directed=True)

        # Add nodes with labels and shapes
        for parent, child in self.tree:
            graph.add_node(parent, shape="circle", label=labels[parent])
            graph.add_node(child, shape="circle", label=labels[child])
            graph.add_edge(parent, child)

        if output_file.endswith(".dot"):
            graph.write(output_file)  # Save as a DOT file

        elif output_file.endswith(".png"):
            graph.draw(output_file, prog="dot", format="png")  # Save as a PNG

    def compute_hz(self, eps=1e-12):

        h_z = -np.sum(self.q_z * np.log(self.q_z + eps), axis=1) / self.q_z.shape[1]
        return h_z

    def compute_hy(self, eps=1e-12):

        h_y = -np.sum(self.q_y * np.log(self.q_y + eps), axis=1) / self.q_y.shape[1]
        return h_y

    # def clade_score(self):
    #     """Return dict """
    #     # cache descendant clone IDs for O(1) lookup later
    #     h_z = self.compute_hz()
    #     h_y = self.compute_hy()
    #     desc_dict = dict(zip(self.genotype_matrix['clone'], self.genotype_matrix["descendants"]))
    #     se = {}
    #     for idx, r in enumerate(self.clones):
    #         C_e = desc_dict[r]       # clone IDs under the edge
    #         C_e_indices = [idx_p for idx_p, p in enumerate(self.clones) if p in C_e ]
    #         # soft weights: probability each item sits somewhere in C_e
    #         w_cells = self.q_z[:, C_e_indices].sum(1)           # shape (n_cells,)
    #         w_snvs  = self.q_y[:, C_e_indices].sum(1)           # same for SNVs

    #         num = (w_cells * h_z).sum() + (w_snvs * h_y).sum()
    #         den = w_cells.sum()       + (w_snvs).sum()
    #         se[r] = 1.0 - num / den
    #     return se
