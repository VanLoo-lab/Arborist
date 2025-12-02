import pygraphviz as pgv
from collections import defaultdict
import itertools
import pandas as pd


def get_nodes(adj_list):
    """
    Given an adjacency list, return all nodes in the graph.

    Parameters
    ----------
    adj_list : dict of int to list of int
        An adjacency list where each key is a node and the value is a list of its
        children.
    Returns
    -------
    set
        A set of all nodes in the graph.
    """
    return set(adj_list.keys()) | set(itertools.chain.from_iterable(adj_list.values()))


def edge_list_to_adj_list(edges):
    """
    Converts a list of edges to an adjacency list.

    Parameters
    ----------
    edges : list of tuple of int
        A list of edges, where each edge is a tuple of two integers representing
        a parent-child relationship.

    Returns
    -------
    dict of int to list of int
        An adjacency list where each key is a node and the value is a list of its
        children.
    """
    adjacency_list = defaultdict(list)
    for parent, child in edges:
        adjacency_list[parent].append(child)
    return adjacency_list


def get_descendants(tree_dict, node):
    """
    Given a tree as a list of (parent, child) edges, return all descendants of a given node.

    Parameters
    ----------
    tree_dict : dict of int to list of int
        A dictionary where each key is a node and the value is a list of its children.
    node : int
        The node for which to find descendants.

    Returns
    -------
    set
        A set of all descendant nodes.
    """

    # Recursive function to collect descendants
    def collect_descendants(n):
        descendants = set(tree_dict.get(n, []))  # Get children (if any)
        for child in tree_dict.get(n, []):
            descendants.update(collect_descendants(child))
        return descendants

    return collect_descendants(node)


def read_trees(file_path, sep=" "):
    """
    Reads tree edges from a file and organizes them into a list of trees.

    Each tree is represented as a list of tuples, where each tuple contains
    two integers representing a parent-child relationship. Trees are separated
    by empty lines or comment lines (lines starting with `#`) in the file.

    Parameters
    ----------
    file_path : str
        The path to the file containing tree edge data. The file should have
        edge delimited integers on each line representing parent-child
        relationships.
    sep : str, optional
        The delimiter between parent and child in the input file (default is ``" "``).

    Returns
    -------
    list of list of tuple of int
        A list of trees, where each tree is a list of tuples. Each tuple
        contains two integers representing a parent-child relationship.

    Notes
    -----
    - Empty lines and lines starting with `#` are treated as separators
      between trees.
    - If the file ends without a separator, the last tree is still included
      in the output.

    Examples
    --------
    Given a file `path/to/file` with the following content:

    ```
    # Tree 1
    1 2
    1 3
    # Tree 2
    1 2
    2 3
    ```



    >>> read_trees("path/to/file")
    [[(1, 2), (1, 3)], [(4, 5), (4, 6)]]
    """
    trees = []
    current_tree = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or "#" in line:  # Skip empty lines and comments
                if current_tree:  # Store previous tree before starting a new one
                    trees.append(current_tree)
                    current_tree = []
                continue

            # Convert space-separated numbers to tuple (parent, child)
            parts = line.split(sep)
            if len(parts) == 2:
                current_tree.append((int(parts[0]), int(parts[1])))

    if current_tree:  # Append last tree if exists
        trees.append(current_tree)

    return trees



def visualize_tree(tree, cell_assignment=None, output_file=None):
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
    for parent, child in tree:
        labels[parent] = str(parent)
        labels[child] = str(child)
    if cell_assignment:
        cell_mapping = defaultdict(list)
        for node, cell in cell_assignment.items():
            cell_mapping[cell].append(node)

        for node, cells in cell_mapping.items():
            labels[node] = f"{node}\n({len(cells)} cells)"

    graph = pgv.AGraph(directed=True)

    # Add nodes with labels and shapes
    for parent, child in tree:
        graph.add_node(parent, shape="circle", label=labels[parent])
        graph.add_node(child, shape="circle", label=labels[child])
        graph.add_edge(parent, child)

    # Add cell assignment information

    # if cell_mapping:
    #     for node, cells in cell_mapping.items():
    #         graph.add_node(f"{node}_cells", shape="none", label=f"{len(cells)} cells", color="none")
    #         graph.add_edge(f"{node}_cells", node, style="dashed", penwidth=1, arrowhead="none")

    if output_file.endswith(".dot"):
        graph.write(output_file)  # Save as a DOT file

    elif output_file.endswith(".png"):
        graph.draw(output_file, prog="dot", format="png")  # Save as a PNG


def ancestor_descendant_set(df):
    ad_set = set()
    for idx, row in df.iterrows():
        u = row["clone"]
        for v in row["descendants"]:
            if u != v:
                ad_set.add((u, v))
    return ad_set


def norm_ad_distance(ad1, ad2):
    diff = ad1.symmetric_difference(ad2)
    union = ad1.union(ad2)
    return len(diff) / (len(union))


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
