from dataclasses import dataclass
import pandas as pd

@dataclass
class TreeFit:
    """
    Solution class to hold the solution to a tree
    """
    tree: list 
    tree_idx: int 
    expected_log_likelihood: float
    q_z: dict
    q_y: dict
    
    def __str__(self):
        return f"{self.tree_idx}: {self.expected_log_likelihood}"
    
    @staticmethod
    def convert_q_to_dataframe(q_dict):
        """
        Convert a nested dictionary {row_id: {column_id: value}} to a DataFrame
        with row_ids as a column named 'id' and column_ids as the remaining columns.
        """
        df = pd.DataFrame.from_dict(q_dict, orient='index')
        df.index.name = "id"
        return df.reset_index()
    
    def q_z_df(self):
        return self.convert_q_to_dataframe(self.q_z)
    
    def q_y_df(self):
        return self.convert_q_to_dataframe(self.q_y)
    
    def __repr__(self):
        return f"TreeFit(tree={self.tree}, tree_idx={self.tree_idx}, expected_log_likelihood={self.expected_log_likelihood})"

    @staticmethod
    def map_assign(q_dict):
        """
        assigns cell to the maximum a posteriori (MAP) clone/cluster
        """
        q_assign = {}
        for val, clone_dict in q_dict.items():
            q_assign[val] = max(clone_dict, key=clone_dict.get)
        
        df = pd.DataFrame(q_assign.items(), columns=["id", "assignment"])
        return df 
    
    def map_assign_z(self):
        """
        assigns cell to the maximum a posteriori (MAP) clone
        """
        return self.map_assign(self.q_z)
    
    def map_assign_y(self):
        """
        assigns snvs to the maximum a posteriori (MAP) cluster
        """
        return self.map_assign(self.q_y)