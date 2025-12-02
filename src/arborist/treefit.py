from dataclasses import dataclass
import pandas as pd
import numpy as np
import pygraphviz as pgv
import networkx as nx
from collections import defaultdict


@dataclass
class TreeFit:
    """
    Solution class to hold the solution for each tree fit by Arborist

    Attributes
    -----------
    tree : list of tuples of int
        Edge list of tree including normal clone.
    tree_idx : int 
        Index identifier of tree in the original input
    elbo : float
        The evidence lower bound (ELBO) computed for the tree.
    q_z : numpy.ndarray
        The inferred approximation to the posterior cell-to-clone label.
    q_y : numpy.ndarray
        The inferred approximation to the posterior SNV-to-cluster label.
    cell_to_idx. : dict[str,int]
        The internal mapping of cell label to index.
    snv_to_idx : dict[str,int]
        The internal mapping of SNV label to index.
    clone_to_idx : dict[str,int]
        The internal mapping of clone label to index.
    cluster_to_idx : dict[str,int]
        The internal mapping of SNV cluster label to index.

    """

    tree: list   
    tree_idx: int 
    elbo: float 
    q_z: np.ndarray 
    q_y: np.ndarray #the inferred approximation to the posterior SNV-to-cluster label
    cell_to_idx: dict #internal cell label to index 
    snv_to_idx: dict #intenal SNV label to index
    clone_to_idx: dict #internal clone label to index
    cluster_to_idx: dict #interal SNV cluster label to index 

    def __post_init__(self):
        self.idx_to_clone = {v: k for k, v in self.clone_to_idx.items()}
        self.idx_to_cluster = {v: k for k, v in self.cluster_to_idx.items()}

    def __str__(self):
        mystr = f"Tree {self.tree_idx}\nELBO: {self.elbo:.2f}\n"
        for u, v in self.tree:
            mystr += f" {u}->{v}\n"
        return mystr

    def _convert_q_to_dataframe(self, q, row_dict, col_dict, prefix="clone"):
        """
        converts q to a dataframe
        """
        df = pd.DataFrame(q)
        df.columns = [f"{prefix}_{col_dict[i]}" for i in range(q.shape[1])]
        df.index.name = "id"
        df.reset_index(inplace=True)
        label_dict = {val: key for key, val in row_dict.items()}
        df["id"] = df["id"].map(label_dict)

        return df

    def q_z_df(self):
        """
        Converts q_z numpy array to pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
            A dataframe containing the inferred cell-to-clone label posterior distribution. 

        """
        df = self._convert_q_to_dataframe(self.q_z, self.cell_to_idx, self.idx_to_clone, prefix="clone")

        return df

    def q_y_df(self):
        """
        Converts q_z numpy array to pandas.DataFrame
        
        Returns
        -------
        pandas.DataFrame
            A dataframe containing the inferred SNV-to-cluster label posterior distribution. 

        """
        df = self._convert_q_to_dataframe(self.q_y, self.snv_to_idx, self.idx_to_cluster, prefix="cluster")

        return df

    def __repr__(self):
        return f"TreeFit(tree={self.tree}, tree_idx={self.tree_idx}, expected_log_likelihood={self.elbo})"

    @staticmethod
    def _map_assign(q, mydict, assign_dict):
        """
        assigns cell to the maximum a posteriori (MAP) clone/cluster
        """
        q_assign = q.argmax(axis=1)
        q_assign = [assign_dict[val] for val in q_assign]

        q_df = pd.DataFrame(q_assign, columns=["assignment"])
        q_df.index.name = "id"
        q_df.reset_index(inplace=True)
        label_dict = {val: key for key, val in mydict.items()}
        q_df["id"] = q_df["id"].map(label_dict)

        return q_df

    def map_assign_z(self):
        """
        Assigns cell to the maximum a posteriori (MAP) clone

        Returns
        -------
        pandas.DataFrame
            A dataframe with columns ['id', 'assignment'] providing the cell-to-clone MAP assignment.

        """
        return self._map_assign(self.q_z, self.cell_to_idx, self.idx_to_clone)

    def map_assign_y(self):
        """
        Assigns SNVs to the maximum a posteriori (MAP) cluster

        Returns
        -------
        pandas.DataFrame
            A dataframe with columns ['id', 'assignment'] providing the SNV-to-cluster MAP assignment
        """
        return self._map_assign(self.q_y, self.snv_to_idx, self.idx_to_cluster)

    def save_tree(self, fname, sep=" "):
        """
        Write the tree as a flat file in the same format as in the input style.

        Parameters
        ----------
        fname : str
            The name of the output file to write the tree.
        sep : str
            The delimiter to use to separate parent and child (default is " ").
        """
        with open(fname, "w+") as file:
            file.write(f"{len(self.tree)} #edges tree 0\n")
            for u, v in self.tree:
                file.write(f"{u}{sep}{v}\n")

    def visualize_tree(self,  output_file):
        """
        Visualizes a tree using Graphviz.

        Parameters
        ----------
        output_file : str
            The path to save the visualization. 


        Notes
        -----
        If a '.dot' extension is passed, the file will be saved as a dot file. Otherwise,
        the format will be autodetected by extension, i.e., png or pdf. 


        """     
   
        labels = defaultdict(str)


        for parent, child in self.tree:
            labels[parent] = f"{parent}"
            labels[child] = f"{child}"



        graph = pgv.AGraph(directed=True)

        # Add nodes with labels and shapes
        for parent, child in self.tree:
            graph.add_node(parent, shape="circle", label=labels[parent])
            graph.add_node(child, shape="circle", label=labels[child])
            graph.add_edge(parent, child)

        if output_file.endswith(".dot"):
            graph.write(output_file)  # Save as a DOT file

        else:
            graph.draw(output_file, prog="dot")  # Save as a PNG

    def _save_genotypes(self, fname, x=1, y=1):
        """
        Write the genotypes of each clone to a file. Default copy numbers (x,y) are (1,1)
        """
        snv_to_cluster_df = self.map_assign_y()
        snv_to_cluster = dict(
            zip(snv_to_cluster_df["id"], snv_to_cluster_df["assignment"])
        )
        tree = nx.DiGraph(self.tree)
        root = [n for n in tree if len(list(tree.predecessors(n))) == 0][0]
        geno_dict = {n: {} for n in tree}

        for node in tree:
            for cluster in tree:
                if cluster != root:
                    if node == cluster or tree.has_predecessor(node, cluster):
                        geno_dict[node][cluster] = 1
                    else:
                        geno_dict[node][cluster] = 0
                else:
                    geno_dict[node][cluster] = 0
        geno_list = []

        for n in tree:
            for j, clust in snv_to_cluster.items():
                geno_list.append([n, j, x, y, geno_dict[n][clust], 0, 1])
        geno_df = pd.DataFrame(
            geno_list, columns=["node", "snv", "x", "y", "xbar", "ybar", "segment"]
        )
        geno_df.to_csv(fname, index=False)

    def cell_entropy(self, eps=1e-12):
        """
        Computes the entropy of each cell assignment from the approximate variational posterior q_z
        
        Returns
        -------
        pandas.DataFrame
            A dataframe with columns ['id', 'entropy'] containing the 
            entropy of the inferred posterior distribution for each cell.

        """

        row_dict = self.cell_to_idx
        h_z = self._compute_hz(eps)
        df = pd.DataFrame(h_z)
        df.columns = ["entropy"]
        df.index.name = "id"
        df.reset_index(inplace=True)
        label_dict = {val: key for key, val in row_dict.items()}
        df["id"] = df["id"].map(label_dict)

        return df

    def snv_cluster_entropy(self, eps=1e-12):
        """
        Computes the entropy of each SNV assignment from the approximate variational posterior q_y
        
        Returns
        -------
        pandas.DataFrame
            A dataframe with columns ['id', 'entropy'] containing the 
            entropy of the inferred posterior distribution for each SNV.
        """
        row_dict = self.snv_to_idx
        h_y = self._compute_hy(eps)
        assert h_y.shape[0] == len(row_dict)
        df = pd.DataFrame(h_y)
        df.columns = ["entropy"]
        df.index.name = "id"
        df.reset_index(inplace=True)
        label_dict = {val: key for key, val in row_dict.items()}
        df["id"] = df["id"].map(label_dict)

        return df

    def _compute_hz(self, eps=1e-12):
        """
        Helper function to compute cell entropy
        """
        h_z = -np.sum(self.q_z * np.log(self.q_z + eps), axis=1) / self.q_z.shape[1]
        return h_z

    def _compute_hy(self, eps=1e-12):
        """
        Helper function to compute SNV entropy
        """
        h_y = -np.sum(self.q_y * np.log(self.q_y + eps), axis=1) / self.q_y.shape[1]
        return h_y
