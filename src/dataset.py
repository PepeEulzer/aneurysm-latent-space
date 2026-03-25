from torch.utils.data import Dataset
import torch
import os
import csv
import random
from src.utils import read_obj 

class AneurysmDataset(Dataset):
    def __init__(self, data_dir='./datasets/aneursym_objs_716/', 
                 rupture_labels_file_path="./datasets/rupture_labels.csv", 
                 split="train", 
                 train_split_percentage=0.8, 
                 seed=42):
        
        self.split = split
        self.data_dir = data_dir
        
        # 1. Get all object files and sort them to ensure consistent ordering across OS
        all_file_names = sorted([f for f in os.listdir(data_dir) if f.endswith(".obj")])
        
        # 2. Deterministic Shuffle (Crucial for preventing Train/Val overlap)
        # We always shuffle with the same seed, regardless of split, to ensure 
        # the 80/20 split is perfectly identical across both dataloaders.
        random.seed(seed)
        random.shuffle(all_file_names)
        
        # 3. Split the file names based on the split parameter
        split_size = int(len(all_file_names) * train_split_percentage)
        
        if self.split == "train":
            self.file_names = all_file_names[:split_size]
        elif self.split == "val":
            self.file_names = all_file_names[split_size:]
        else:
            raise ValueError("Split must be 'train' or 'val'")

        # 4. Load CSV labels into a dictionary
        rupture_dict = {}
        with open(rupture_labels_file_path, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                rupture_dict[row["dataset"]] = row["status"]
        
        # 5. Pre-load Point clouds and Labels ONLY for this specific split
        self.point_clouds = []
        self.class_labels = []
        
        # Track statistics for logging
        counts = {0: 0, 1: 0, 2: 0} # 0: unruptured, 1: ruptured, 2: unknown
        
        print(f"Loading {len(self.file_names)} files for {self.split} split...")
        
        for file_name in self.file_names:
            file_path = os.path.join(self.data_dir, file_name)
            
            # Load point cloud
            verts, _ = read_obj(file_path) # Assuming read_obj is defined elsewhere
            self.point_clouds.append(verts)
            
            # Assign Label
            status = rupture_dict.get(file_name, "unknown")
            
            if status == "ruptured":
                label = 1
            elif status == "unruptured" or "other":
                label = 0
            else: # "unknown", or missing from CSV
                label = 2
                
            self.class_labels.append(label)
            counts[label] += 1
            
        print(f"[{self.split.upper()} SET] Ruptured: {counts[1]} | Unruptured: {counts[0]} | Unknown/Other: {counts[2]}")

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        return self.point_clouds[idx], self.class_labels[idx]


class AneurysmDatasetLOO(Dataset):
    def __init__(self, data_dir, rupture_labels_file_path, split="train", leave_out_source=None):
        """
        split: "train" (loads all sources EXCEPT leave_out_source) 
               "val" (loads ONLY the leave_out_source)
        leave_out_source: String representing the hospital/source to hold out.
        """
        self.split = split
        self.data_dir = data_dir
        self.leave_out_source = leave_out_source
        
        # 1. Load CSV to map filename -> {status, source}
        rupture_dict = {}
        with open(rupture_labels_file_path, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                rupture_dict[row["dataset"]] = {
                    "status": row["status"],
                    "source": row["source"].strip() if row["source"] else "unknown"
                }
        
        # 2. Get all object files
        all_file_names = sorted([f for f in os.listdir(data_dir) if f.endswith(".obj")])
        
        # 3. Filter files based on the Leave-One-Out logic
        self.file_names = []
        for f in all_file_names:
            f_source = rupture_dict.get(f, {}).get("source", "unknown")
            
            if self.split == "val" and f_source == self.leave_out_source:
                self.file_names.append(f)
            elif self.split == "train" and f_source != self.leave_out_source:
                self.file_names.append(f)

        # 4. Pre-load Point clouds and Labels
        self.point_clouds = []
        self.class_labels = []
        counts = {0: 0, 1: 0, 2: 0} 
        
        print(f"Loading {len(self.file_names)} files for LOO {self.split.upper()} split (Holdout: {self.leave_out_source})...")
        
        for file_name in self.file_names:
            file_path = os.path.join(self.data_dir, file_name)
            
            # Assuming read_obj is imported/available from your utils
            verts, _ = read_obj(file_path) 
            self.point_clouds.append(verts)
            
            # Assign Label
            status = rupture_dict.get(file_name, {}).get("status", "unknown")
            if status == "ruptured": 
                label = 1
            elif status == "unruptured" or "other": 
                label = 0
            else: 
                label = 2
                
            self.class_labels.append(label)
            counts[label] += 1
            
        print(f"[{self.split.upper()} SET] Ruptured: {counts[1]} | Unruptured: {counts[0]} | Unknown: {counts[2]}")

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        return self.point_clouds[idx], self.class_labels[idx]
