from os import path

import GEUtils
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
        mesh = cls.simplify_mesh(mesh_path, subsample)
        return cls(mesh)

    @staticmethod
    def simplify_mesh(mesh_path, subsample):
        """
        Load and simplify a mesh, then normalize surface area to 1.
        """
        mesh = trimesh.load(mesh_path)
        simplified_mesh = mesh.simplify_quadric_decimation(percent=1 - subsample)

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

        # Project global coordinates into this local gauge (GET original feature map)
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
    RADIUS = 0.1  # Radius to build neighborhoods
    MAX_NEIGH = 50  # Max neighbors for each vertex

    base = "../data/meshes/"
    out_dir = "../data/SHREC11/processed/"

    # Find meshes that still need processing
    to_process = [i for i in range(0, 600) if not path.exists(f"{out_dir}T{i}.pt")]

    for idx in tqdm(to_process):
        mesh_path = f"{base}T{idx}.off"

        # Load & simplify (if this fails, let exception propagate)
        preprocessor = MeshPreprocessor.from_file(mesh_path, subsample=SUBSAMPLE)

        # Try computing data; if it fails try cleaning once then recompute
        try:
            per_vertex = preprocessor.compute_log_and_ptransport(
                radius=RADIUS, max_neighbors=MAX_NEIGH
            )
        except Exception:
            # attempt a single mesh cleaning pass then re-run
            preprocessor.clean_mesh()
            per_vertex = preprocessor.compute_log_and_ptransport(
                radius=RADIUS, max_neighbors=MAX_NEIGH
            )

        N = len(per_vertex)  # number of vertices in the processed mesh

        # Allocate fixed-size tensors
        features = torch.zeros((N, 3), dtype=torch.float32)
        neighbors = torch.full((N, MAX_NEIGH), -1, dtype=torch.long)
        u_q = torch.zeros((N, MAX_NEIGH, 2), dtype=torch.float32)
        g_qp = torch.zeros((N, MAX_NEIGH), dtype=torch.float32)
        mask = torch.zeros((N, MAX_NEIGH), dtype=torch.bool)

        for v, d in enumerate(per_vertex):
            q_idx = d["q_indices"]
            k = min(len(q_idx), MAX_NEIGH)

            features[v] = torch.from_numpy(d["features"])
            neighbors[v, :k] = torch.from_numpy(q_idx)
            u_q[v, :k] = torch.from_numpy(d["u_q"])
            g_qp[v, :k] = torch.from_numpy(d["g_qp"])
            mask[v, :k] = True

        # Precompute parallel transport matrices for the mesh.
        # This call may raise if shapes or values are unexpected; do not swallow exceptions.
        r2r = GEUtils.RegularToRegular(N)
        # g_qp is a torch tensor of shape (N, MAX_NEIGH)
        parallel_transport = r2r.extended_regular_representation(g_qp)

        save_dict = {
            "features": features,
            "neighbors": neighbors,
            "u_q": u_q,
            "g_qp": g_qp,
            "mask": mask,
            "parallel_transport": parallel_transport,
        }

        torch.save(save_dict, f"{out_dir}T{idx}.pt")
        preprocessor.mesh.export(f"{out_dir}T{idx}.off")

    print("Preprocessing completed.")
