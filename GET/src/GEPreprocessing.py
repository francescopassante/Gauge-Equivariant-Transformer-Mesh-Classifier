from os import path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import potpourri3d as pp3d
import torch
import trimesh
from tqdm import tqdm


class MeshPreprocessor:
    def __init__(self, mesh):
        self.mesh = mesh

    @classmethod
    def from_file(cls, mesh_path, subsample):
        mesh = cls.simplify_mesh(cls, mesh_path, subsample)
        return cls(mesh)

    def __str__(self):
        return f"MeshPreprocessor(mesh with {len(self.mesh.vertices)} vertices and {len(self.mesh.faces)} faces)"

    def simplify_mesh(self, mesh_path, subsample):
        # Load the mesh
        mesh = trimesh.load(mesh_path)

        # Subsample the mesh using quadric decimation to reduce the number of vertices while preserving the overall shape
        simplified_mesh = mesh.simplify_quadric_decimation(percent=1 - subsample)

        #  Normalize surface area to 1
        area = simplified_mesh.area
        if area > 0:
            simplified_mesh.apply_scale(1 / np.sqrt(area))

        return simplified_mesh

    def compute_geodesic_neighborhood(self, p_idx, radius):

        solver = pp3d.MeshHeatMethodDistanceSolver(self.mesh.vertices, self.mesh.faces)

        # 3. Compute distances from a source vertex p (index p_idx)
        distances = solver.compute_distance(p_idx)

        # 4. Select vertices within the RADIUS geodesic radius
        neighbor_indices = np.where(distances <= radius)[0]
        return neighbor_indices

    def compute_log_and_ptransport(self, radius, max_neighbors):
        vertices = self.mesh.vertices.astype(np.float32)
        num_vertices = len(vertices)

        dist_solver = pp3d.MeshHeatMethodDistanceSolver(vertices, self.mesh.faces)
        vector_solver = pp3d.MeshVectorHeatSolver(vertices, self.mesh.faces)

        # Get the exact basis used by the solver for log-maps and transport
        basis_x, basis_y, normals = vector_solver.get_tangent_frames()

        # Project global coordinates into this local gauge: [<p,x>, <p,y>, <p,n>]. This is the GET original feature map
        local_x = np.einsum("ij,ij->i", vertices, basis_x)[:, np.newaxis]
        local_y = np.einsum("ij,ij->i", vertices, basis_y)[:, np.newaxis]
        local_z = np.einsum("ij,ij->i", vertices, normals)[:, np.newaxis]
        features = np.hstack([local_x, local_y, local_z]).astype(np.float32)

        data = []

        # If you must do all-to-all transport, the most efficient way in
        # potpourri3d is to iterate and use the pre-factored back-substitution.
        for i in range(num_vertices):
            # 1. Distances and Log Map (Fast because solver is pre-factored)
            dists = dist_solver.compute_distance(i)
            log_map = vector_solver.compute_log_map(i)  # (N, 2)

            # 2. Get neighbors
            neighbor_indices = np.where(dists <= radius)[0]
            neighbor_indices = neighbor_indices[neighbor_indices != i]

            # Caps the number of neighbors to max_neighbors, keeping the closest ones
            if len(neighbor_indices) > max_neighbors:
                neighbor_indices = neighbor_indices[
                    np.argsort(dists[neighbor_indices])
                ][:max_neighbors]

            # 3. Parallel Transport
            # We compute the transport from 'i' to its neighbors here.
            # Note: transport(i->q) is the inverse of transport(q->i).
            # We transport the basis vector [1,0] from center i to all neighbors.
            transport_from_center = vector_solver.transport_tangent_vector(
                i, [1.0, 0.0]
            )

            # Extract angles for neighbors
            v_neighbors = transport_from_center[neighbor_indices]
            g_pq = np.arctan2(v_neighbors[:, 1], v_neighbors[:, 0])

            # g_qp (neighbor to center) is simply -g_pq
            g_qp = -g_pq

            data.append(
                {
                    "features": features[i],
                    "q_indices": neighbor_indices.astype(np.int32),
                    "u_q": log_map[neighbor_indices].astype(np.float32),
                    "g_qp": g_qp.astype(np.float32),
                }
            )

        return data

    def clean_mesh(self):
        """Clean the mesh by removing duplicated vertices, degenerate triangles, and non-manifold edges using Open3D and Trimesh."""
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(self.mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(self.mesh.faces)

        o3d_mesh.remove_duplicated_vertices()
        o3d_mesh.remove_duplicated_triangles()
        o3d_mesh.remove_degenerate_triangles()
        o3d_mesh.remove_non_manifold_edges()

        vertices = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles)
        clean_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

        clean_mesh.fill_holes()

        self.mesh = clean_mesh

    # Function to plot the neighbors of vertex 0, debug purposes:
    def plot_neighbors(self, p_idx, distance, ax):
        neighbor_indices = self.compute_geodesic_neighborhood(p_idx, distance)

        ax.scatter(
            self.mesh.vertices[neighbor_indices, 0],
            self.mesh.vertices[neighbor_indices, 1],
            self.mesh.vertices[neighbor_indices, 2],
            color="red",
            s=10,
        )

        # Plot the source vertex in blue
        ax.scatter(
            self.mesh.vertices[p_idx, 0],
            self.mesh.vertices[p_idx, 1],
            self.mesh.vertices[p_idx, 2],
            color="k",
            s=50,
        )


if __name__ == "__main__":
    SUBSAMPLE = 0.2
    RADIUS = 0.2
    MAX_NEIGH = 300

    base = "../data/SHREC11_test_database_new/"
    paths = [i for i in range(0, 600) if not path.exists(f"../data/processed/T{i}.pt")]

    for j, filename in enumerate(tqdm(paths)):
        preprocessor = MeshPreprocessor.from_file(
            base + f"T{filename}.off", subsample=SUBSAMPLE
        )
        try:
            data = preprocessor.compute_log_and_ptransport(
                radius=RADIUS, max_neighbors=MAX_NEIGH
            )
        except Exception:
            preprocessor.clean_mesh()
            try:
                data = preprocessor.compute_log_and_ptransport(
                    radius=RADIUS, max_neighbors=MAX_NEIGH
                )
            except Exception as e:
                print(f"Failed to process T{filename}.off after cleaning: {e}")
                continue

        N = len(data)  # number of vertices

        # Preallocate tensors
        features = torch.zeros((N, 3), dtype=torch.float32)  # local features
        neighbors = torch.full((N, MAX_NEIGH), -1, dtype=torch.long)  # neighbor indices
        u_q = torch.zeros((N, MAX_NEIGH, 2), dtype=torch.float32)  # 2D vectors
        g_qp = torch.zeros((N, MAX_NEIGH), dtype=torch.float32)  # cos/sin angles
        mask = torch.zeros((N, MAX_NEIGH), dtype=torch.bool)  # valid neighbors mask

        # Fill tensors
        for i, d in enumerate(data):
            q_indices = d["q_indices"]
            n = min(
                len(q_indices), MAX_NEIGH
            )  # number of neighbors (capped at MAX_NEIGH)

            u = d["u_q"]
            g = d["g_qp"]

            # store features
            features[i] = torch.from_numpy(d["features"])
            # store neighbor indices
            neighbors[i, :n] = torch.from_numpy(q_indices)
            # store vectors
            u_q[i, :n] = torch.from_numpy(u)
            # store angles
            g_qp[i, :n] = torch.from_numpy(g)
            # mask
            mask[i, :n] = True

        # Save as a PyTorch file
        torch.save(
            {
                "features": features,
                "neighbors": neighbors,
                "u_q": u_q,
                "g_qp": g_qp,
                "mask": mask,
            },
            f"../data/processed/T{filename}.pt",
        )
        # Save the preprocessed mesh as well, for reference (optional)
        preprocessor.mesh.export(f"../data/processed/T{filename}.off")
