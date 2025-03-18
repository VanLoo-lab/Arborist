import pygraphviz as pgv
from collections import defaultdict


def read_tree_edges_conipher(file_path):
    """
    Reads tree edges from a file and organizes them into a list of trees.

    Each tree is represented as a list of tuples, where each tuple contains
    two integers representing a parent-child relationship. Trees are separated
    by empty lines or comment lines (lines starting with `#`) in the file.

    Parameters
    ----------
    file_path : str
        The path to the file containing tree edge data. The file should have
        space-separated integers on each line representing parent-child
        relationships.

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
    Given a file with the following content:

    ```
    # Tree 1
    1 2
    1 3

    # Tree 2
    4 5
    4 6
    ```

    Calling `read_tree_edges_conipher("path/to/file")` will return:

    >>> read_tree_edges_conipher("path/to/file")
    [[(1, 2), (1, 3)], [(4, 5), (4, 6)]]
    """
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
    """
    Reads tree edges from a file and organizes them into a list of trees.

    Each tree is represented as a list of tuples, where each tuple contains
    two integers representing a parent-child relationship. Trees are separated
    in the file by lines starting with a specified header prefix.

    Parameters
    ----------
    file_path : str
        The path to the file containing the tree edge data.
    header_prefix : str, optional
        The prefix that indicates the start of a new tree in the file.
        Default is "backbone tree".

    Returns
    -------
    list of list of tuple of int
        A list of trees, where each tree is a list of tuples. Each tuple
        represents a parent-child relationship as (parent, child).

    Notes
    -----
    - Lines starting with "#" or empty lines are ignored.
    - The file is expected to have space-separated integers for parent-child
      relationships.
    - If the file ends without a header for a new tree, the last tree is
      automatically added to the result.

    Examples
    --------
    Given a file with the following content:

    ```
    # Example tree file
    backbone tree 1
    1 2
    1 3
    backbone tree 2
    4 5
    4 6
    ```

    Calling the function:

    >>> read_tree_edges_sapling("example.txt")
    [[(1, 2), (1, 3)], [(4, 5), (4, 6)]]
    """
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
    labels = {n: str(n) for n in tree}
    if cell_assignment:
        cell_mapping = defaultdict(list)
        for node, cell in cell_assignment.items():
            cell_mapping[cell].append(node)
        
        for node , cells in cell_mapping.items():
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