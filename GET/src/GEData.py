from os import path

import GEUtils
import torch
from torch.utils.data import Dataset


class MeshDataset(Dataset):
    def __init__(self, mesh_directory, labels_file, N):
        # Ensure directory path ends with a slash
        self.base_path = (
            mesh_directory if mesh_directory.endswith("/") else mesh_directory + "/"
        )

        # Collect existing processed files named T{idx}.pt (original dataset up to 600)
        self.filenumbers = [
            i for i in range(600) if path.exists(f"{self.base_path}T{i}.pt")
        ]

        # Helper to compute extended regular representation when needed
        self.r2r = GEUtils.RegularToRegular(N)

        # Load labels file (expected format: blocks of 21 lines -> class name + 20 indices)
        with open(labels_file) as f:
            lines = [line.strip() for line in f if line.strip()]

        labels = [0] * 600
        for class_idx, start in enumerate(range(0, len(lines), 21)):
            block = lines[start : start + 21]
            for idx_str in block[1:]:
                labels[int(idx_str)] = class_idx

        self.labels = labels

    def __len__(self):
        return len(self.filenumbers)

    def __getitem__(self, idx):
        file_index = self.filenumbers[idx]
        data = torch.load(f"{self.base_path}T{file_index}.pt")

        return {
            "x": data["features"],
            "neighbors": data["neighbors"],
            "mask": data["mask"],
            "parallel_transport": data["parallel_transport"],
            "rel_pos": data["u_q"],
            "label": self.labels[file_index],
        }
