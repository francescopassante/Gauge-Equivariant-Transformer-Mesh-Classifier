from os import path

import GEUtils
import torch
from torch.utils.data import Dataset


class MeshDataset(Dataset):
    def __init__(self, mesh_directory, labels_file, N, filenumbers=None):
        self.base_path = mesh_directory
        self.N = N
        self._r2r = GEUtils.RegularToRegular(N)

        if filenumbers is not None:
            # Use the provided list directly (e.g. when resuming a session)
            self.filenumbers = list(filenumbers)
        else:
            # Scan directory for all available processed files
            self.filenumbers = [
                i for i in range(600) if path.exists(f"{self.base_path}T{i}.pt")
            ]

        # Load labels file
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
        data = torch.load(f"{self.base_path}T{file_index}.pt", weights_only=False)

        parallel_transport_matrices = self._r2r.extended_regular_representation(
            data["g_qp"]
        )

        return {
            "x": data["features"],
            "neighbors": data["neighbors"],
            "mask": data["mask"],
            "parallel_transport_matrices": parallel_transport_matrices,
            "rel_pos": data["u_q"],
            "label": self.labels[file_index],
            "filenumber": file_index,
        }
