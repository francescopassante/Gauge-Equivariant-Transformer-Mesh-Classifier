from os import path

import numpy as np
import open3d as o3d
import potpourri3d as pp3d
import torch
import trimesh
from tqdm import tqdm


class MeshPreprocessor:
    """
    A class for preprocessing mesh data.
    """

    def __init__(self, mesh):
        self.mesh = mesh

    @classmethod
    def from_file(cls, mesh_path, subsample):
        mesh = cls.simplify_mesh(cls, mesh_path, subsample)
        return cls(mesh)

    def __str__(self):
        return f"MeshPreprocessor(mesh with {len(self.mesh.vertices)} vertices and {len(self.mesh.faces)} faces)"

    def simplify_mesh(self, mesh_path, subsample):
        """
        Simplify the mesh by subsampling it and scaling the sufrace area to 1.
        """
        # Load the mesh
        mesh = trimesh.load(mesh_path)

        # Subsample the mesh using quadric decimation to reduce the number of vertices
        simplified_mesh = mesh.simplify_quadric_decimation(percent=1 - subsample)

        #  Normalize surface area to 1
        area = simplified_mesh.area
        if area > 0:
            simplified_mesh.apply_scale(1 / np.sqrt(area))

        return simplified_mesh

    def compute_log_and_ptransport(self, radius, max_neighbors):
        """
        Efficiently compute the log map and parallel transport angle for each vertex to its neighbors within a given radius,
        capping the number of neighbors to max_neighbors.
        """

        vertices = self.mesh.vertices.astype(np.float32)
        num_vertices = len(vertices)

        # Prepares the solvers
        dist_solver = pp3d.MeshHeatMethodDistanceSolver(vertices, self.mesh.faces)
        vector_solver = pp3d.MeshVectorHeatSolver(vertices, self.mesh.faces)

        # Get the exact basis used by the solver for log-maps and transport. Used to compute features
        basis_x, basis_y, normals = vector_solver.get_tangent_frames()

        # Project global coordinates into this local gauge: [<p,x>, <p,y>, <p,n>]. This is the GET original feature map
        local_x = np.einsum("ij,ij->i", vertices, basis_x)[:, np.newaxis]
        local_y = np.einsum("ij,ij->i", vertices, basis_y)[:, np.newaxis]
        local_z = np.einsum("ij,ij->i", vertices, normals)[:, np.newaxis]
        features = np.hstack([local_x, local_y, local_z]).astype(np.float32)

        # Array to hold features, neighbors, log maps, parallel transport angles.
        data = []

        for i in range(num_vertices):
            # Distances and Log Map
            dists = dist_solver.compute_distance(i)
            log_map = vector_solver.compute_log_map(i)  # (N, 2)

            # Finds neighbors of vertex i (excluding itself)
            neighbor_indices = np.where(dists <= radius)[0]
            neighbor_indices = neighbor_indices[neighbor_indices != i]

            # Caps the number of neighbors to max_neighbors, keeping the closest ones
            if len(neighbor_indices) > max_neighbors:
                neighbor_indices = neighbor_indices[
                    np.argsort(dists[neighbor_indices])
                ][:max_neighbors]

            # This transports a standard vector [1,0] from vertex i to all other vertices in the mesh
            transport_from_center = vector_solver.transport_tangent_vector(
                i, [1.0, 0.0]
            )

            # This only keeps the transported vectors from i to the neighbors
            v_neighbors = transport_from_center[neighbor_indices]

            # This computes the angle
            g_pq = np.arctan2(v_neighbors[:, 1], v_neighbors[:, 0])

            # To go from neighbor to i, it's the inverse of from i to neighbor
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


if __name__ == "__main__":
    SUBSAMPLE = 0.2  # Amount of subsampling (lower = less points)
    RADIUS = 0.2  # Radius to build neighborhoods
    MAX_NEIGH = 300  # Max neighbors for each vertex

    base = "../data/SHREC11_test_database_new/"
    paths = [i for i in range(0, 600) if not path.exists(f"../data/processed/T{i}.pt")]

    for j, filename in enumerate(tqdm(paths)):
        # Initializes the preprocessor
        preprocessor = MeshPreprocessor.from_file(
            base + f"T{filename}.off", subsample=SUBSAMPLE
        )

        # Subsampling often introduces non-manifold structures, we try to clean them up.
        # Most of the meshes are fine with this process, however 7 of them ([5, 31, 100, 130, 324, 426, 533]) don't get cleaned up
        # For these ones, i subsample at 10%, somehow it works better for them. Only 1 (130) keeps being broken, i remove it.
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

        # Preallocate tensors, useful to have this structure for parallel processing
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

            # Store features
            features[i] = torch.from_numpy(d["features"])
            # Store neighbor indices
            neighbors[i, :n] = torch.from_numpy(q_indices)
            # Store vectors
            u_q[i, :n] = torch.from_numpy(u)
            # Store angles
            g_qp[i, :n] = torch.from_numpy(g)
            # Store mask
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
