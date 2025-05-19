from dataclasses import dataclass
import pandas as pd
import numpy as np 

@dataclass
class TreeFit:
    """
    Solution class to hold the solution to a tree
    """
    tree: list 
    tree_idx: int 
    expected_log_likelihood: float
    q_z: np.ndarray
    q_y: np.ndarray
    cell_to_idx: dict
    snv_to_idx: dict
    clones: list
    
    def __str__(self):
        return f"{self.tree_idx}: {self.expected_log_likelihood}"
    

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
        return f"TreeFit(tree={self.tree}, tree_idx={self.tree_idx}, expected_log_likelihood={self.expected_log_likelihood})"

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
    
    def save_tree(self,fname):
        pass 
        with open(fname, "w+") as file:
            file.write("# tree\n")
            for u,v in self.tree:
                file.write(f"{u},{v}\n")
        