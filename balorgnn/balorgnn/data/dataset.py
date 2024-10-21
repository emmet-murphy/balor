
import torch
from torch_geometric.data import Dataset, Data
import os.path as osp
import glob

class CustomData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "cfg_edge_index" or key == "bb_id_list":
            return self.num_bbs
        if key == "bb_batch":
            return 1
        if 'index' in key:
            return self.num_nodes
        else:
            return 0
    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'index' in key:
            return 1
        else:
            return 0
        

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        if not data_dir.endswith("/"):
            data_dir = data_dir + "/"
        self.data_dir = data_dir
        print(self.data_dir)
        super().__init__()

    @property
    def processed_file_names(self):
        return glob.glob(f"{self.data_dir}*.pt")
    
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.data_dir, f'data_{idx}.pt'))
        return data