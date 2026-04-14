from os import path

import GEUtils
import torch
from torch.utils.data import Dataset


class MeshDataset(Dataset):
    def __init__(self, mesh_directory, labels_file, N):
        self.base_path = mesh_directory
        self.filenumbers = [
            i for i in range(600) if path.exists(f"{mesh_directory}T{i}.pt")
        ]
        self.r2r = GEUtils.RegularToRegular(N)

        # Loads labels
        with open(labels_file) as f:
            lines = [line.strip() for line in f if line.strip()]
        # Each class occupies a block of 21 lines: class name then 20 indices.
        labels = [0] * 600
        for class_idx, start in enumerate(range(0, len(lines), 21)):
            block = lines[start : start + 21]
            for idx_str in block[1:]:
                labels[int(idx_str)] = class_idx

        self.labels = labels

    def __len__(self):
        return len(self.filenumbers)

    def __getitem__(self, idx):

        data = torch.load(f"{self.base_path}T{self.filenumbers[idx]}.pt")
        parallel_transport_matrices = self.r2r.extended_regular_representation(
            data["g_qp"]
        )

        return {
            "x": data["features"],
            "neighbors": data["neighbors"],
            "mask": data["mask"],
            "parallel_transport": parallel_transport_matrices,
            "rel_pos": data["u_q"],
            "label": self.labels[self.filenumbers[idx]],
        }
