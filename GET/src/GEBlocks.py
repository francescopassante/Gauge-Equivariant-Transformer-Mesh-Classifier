import GEUtils
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class GELocalToRegularLinearBlock(nn.Module):
    """
    A linear layer that implements equivariance from the rho_local representation to the regular representation for the cyclic group C_N.
    """

    def __init__(self, N, channels):
        """
        Args:
            N: Dimension of the regular representation (C_N).
            channels: Number of regular output fields.
        """
        super().__init__()
        self.N = N
        utils = GEUtils.LocalToRegular(N)
        W_basis = utils.local_to_regular_basis()
        self.register_buffer(
            "basis", torch.stack(W_basis)
        )  # This registers the basis as a non-learnable buffer
        self.num_basis = self.basis.shape[0]  # Number of basis matrices
        self.channels = channels

        # Learnable coefficients for each basis matrix for each output field
        # Initializing with small random values
        self.weights = nn.Parameter(torch.randn(channels, self.num_basis) * 0.02)

    def forward(self, x):
        """
        Args:
            x: Input features (rho_local) of shape (Batch, Num_Points, 3)
        Returns:
            Output feature fields of shape (Batch, Num_Points, channels * N)
        """
        # 1. Compute the kernel for each field: W = sum(a_i * W_basis_i)
        # Resulting shape: (channels, N, 3)
        combined_kernels = torch.einsum("fk,knm->fnm", self.weights, self.basis)

        # 2. Reshape kernels to a single large weight matrix for efficient computation
        # Shape: (channels, N, 3) -> (channels * N, 3)
        W_final = combined_kernels.view(self.channels * self.N, 3)

        # 3. Apply the linear transformation to the input features
        # (B, P, 3) @ (3, channels*N) -> (B, P, channels*N)
        out = torch.matmul(x, W_final.t()).view(x.shape[0], self.channels, self.N)

        return out


class GESelfAttentionBlock(nn.Module):
    def __init__(self, N, in_channels):
        super().__init__()
        self.N = N
        # self.n_heads = n_heads
        # self.d_k = out_channels // n_heads

        self.in_channels = in_channels

        # Equivariant basis for Query and Key linear maps
        basis = GEUtils.RegularToRegular(N).regular_to_regular_basis()
        self.register_buffer("reg_to_reg_basis", torch.stack(basis))

        # Query and Key coefficients are [in_channels, len_basis] because we use a linear comb of the basis for each channel, then sum
        self.query_coeffs = nn.Parameter(torch.randn(in_channels, len(basis)) * 0.02)
        self.key_coeffs = nn.Parameter(torch.randn(in_channels, len(basis)) * 0.02)

        # The value matrix is given by a second order Taylor expansion in the relative position u.
        # The Taylor coefficients (matrices) must satisfy the equivariance condition in Eqn. (78) of the paper.
        # One finds that the equivariance is satisfied order by order, so there are separate bases for the zero, first, and second order terms.
        # So we allow a linear combination of each basis inside a given order.
        # We learn a linear combination of these basis matrices as the value function W_V(u).

        value_basis = GEUtils.RegularToRegular(N).get_taylor_basis()
        self.register_buffer(
            "value_basis_zero_order", value_basis[0].squeeze(1)
        )  # Questa la reshapo perché a ordine 0 c'è solo 1 matrice, utile dopo quando calcolo i values
        self.register_buffer("value_basis_first_order", value_basis[1])
        self.register_buffer("value_basis_second_order", value_basis[2])

        self.value_matrix_zero_order_params = nn.Parameter(
            torch.randn(in_channels, self.value_basis_zero_order.shape[0]) * 0.02
        )
        self.value_matrix_first_order_params = nn.Parameter(
            torch.randn(in_channels, self.value_basis_first_order.shape[0]) * 0.02
        )
        self.value_matrix_second_order_params = nn.Parameter(
            torch.randn(in_channels, self.value_basis_second_order.shape[0]) * 0.02
        )

    def W_K(self, x):
        # key_coeffs are [in_channels, len_basis]
        # reg_to_reg_basis are [len_basis, N, N] (len_basis = N tra l'altro)
        # W_K is a NxN matrix for each channel -> [in_channels, N, N]

        W_K = torch.einsum("cb, bij -> cij", self.key_coeffs, self.reg_to_reg_basis)
        x = x.view(x.shape[0], self.in_channels, self.N)  # x = [N_v, in_channels, N]
        return torch.einsum("cij, vcj -> vi", W_K, x)

    def W_Q(self, fprime):
        # fprime is [N_v, MAX_NEIGH, in_channels, N]
        # W_Q is [in_channels, N, N]
        W_Q = torch.einsum("cb, bij -> cij", self.query_coeffs, self.reg_to_reg_basis)
        return torch.einsum("cij, vncj -> vni", W_Q, fprime)

    def forward(self, x, neighbors, mask, parallel_transport_matrices, rel_pos_u):
        """
        Args:
            x: [N_v, in_channels * N] - Center features
            neighbors: [N_v, Max_Neighbors] - Indices of neighbors
            mask: [N_v, Max_Neighbors] - Binary mask for valid neighbors
            parallel_transport_matrices: [N_v, Max_Neighbors, N, N] - rho_tilde(theta)
            rel_pos_u: [N_v, Max_Neighbors, 2] - Logarithmic map coordinates u_q
        """
        N_v, chan, n = x.shape

        # 1. Parallel Transport neighbors to center frame
        # x_neighbors shape: [N_v, Max_N, in_channels * N]
        x_neigh = x[neighbors]

        x_neigh = x_neigh.view(N_v, -1, self.in_channels, self.N)

        # Zero-out "fake neighbors" so that x_neigh[v][n] is zero if n > actual number of neighbors for vertex v
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # [N_v, Max_N, 1, 1]
        x_neigh = x_neigh * mask_expanded

        # Apply rho_tilde(theta) to each channel
        f_prime_q = torch.einsum(
            "vnij,vncj->vnci", parallel_transport_matrices, x_neigh
        )

        # 2. Compute Attention Scores
        # print("x shape: ", x.shape)
        K = self.W_K(x)  # .view(N_v, -1, self.n_heads, self.d_k)
        Q = self.W_Q(f_prime_q)

        score = (
            torch.relu(Q + K.unsqueeze(1)).mean(dim=-1).masked_fill(~mask, 0)
        )  # [N_v, Max_Neigh, in_channels]

        score_denominator = score.sum(dim=-1).clamp(min=1e-8)

        attention = score / score_denominator.unsqueeze(-1)

        # 3. Compute Values using Equivariant Kernel W_V(u)
        # W_V(u) = W0 + W1*u1 + W2*u2 ... (Taylor Expansion)

        u_0 = rel_pos_u[..., 0]
        u_1 = rel_pos_u[..., 1]
        u_0_squared = u_0**2
        u_1_squared = u_1**2
        u_0_u_1 = (
            2 * u_0 * u_1
        )  # This 2 factor i think is fundamental, goes back to the SVD solution and form of F for the second order

        # Apply value function to transported features
        # V = W_V(u) * f_prime_q

        # Pre-calcola la matrice W per ogni canale: [in_channels, N, N]
        W0 = torch.einsum(
            "cb, bij -> cij",
            self.value_matrix_zero_order_params,
            self.value_basis_zero_order,
        )
        values = torch.einsum("cij, vncj -> vnci", W0, f_prime_q)

        W1 = torch.einsum(
            "cb, boij -> coij",
            self.value_matrix_first_order_params,
            self.value_basis_first_order,
        )
        values += torch.einsum("coij, vno, vncj -> vnci", W1, rel_pos_u, f_prime_q)

        W2 = torch.einsum(
            "cb, boij -> coij",
            self.value_matrix_second_order_params,
            self.value_basis_second_order,
        )

        u_quad = torch.stack([u_0_squared, u_0_u_1, u_1_squared], dim=-1)

        values += torch.einsum("coij, vno, vncj -> vnci", W2, u_quad, f_prime_q)

        # 4. Aggregation
        out = torch.einsum("vn,vnci->vci", attention, values)  # [N_v, in_channels, N]
        return out


class GEGroupPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.max(dim=-1)[0]  # [N_v, in_channels]


class GEGlobalAveragePooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=0)  # [in_channels]


if __name__ == "__main__":
    # I want to check that if i rotate an input (x,y,z) then apply the layer i get a permutation of the output fields:
    def check_equivariance_l2r():
        def rotate_input(x, theta):
            # Rotate around the z-axis by angle theta
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rotation_matrix = torch.tensor(
                [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]],
                dtype=torch.float32,
            )
            return x @ rotation_matrix.t()

        equivariant_layer = GELocalToRegularLinearBlock(N=9, channels=12)

        # Rotate the input by 2pi/9 (the angle corresponding to the cyclic group C9) and check the output
        theta = torch.tensor(2 * np.pi / 9, dtype=torch.float32)

        input = torch.randn(1, 1, 3)  # Original input
        rotated_input = rotate_input(input, theta)

        output = equivariant_layer(input).view(1, 1, 12, 9)
        rotated_output = equivariant_layer(rotated_input).view(1, 1, 12, 9)

        print(output[0][0][3])  # Should be (1, 1, 12, 9)
        print(rotated_output[0][0][3])  # Should be (1, 1, 12, 9)

    def check_equivariance_sa(N, channels):
        # load the data
        path = "../data/processed/T3.pt"
        data = torch.load(path, map_location="cpu")
        x = data["features"]  # (N_v, 3)
        neighbors = data["neighbors"]  # (N_v, Max_Neighbors)
        parallel_transport_angles = data["g_qp"]  # (N_v, Max_Neighbors, N, N)
        rel_pos_u = data["u_q"]  # (N_v, Max_Neighbors, 2)
        mask = data["mask"]  # (N_v, Max_Neighbors)

        # (N_v, in_channels * N)
        l2rBlock = GELocalToRegularLinearBlock(N, channels=channels)

        r2r = GEUtils.RegularToRegular(N)
        parallel_transport_matrices = r2r.extended_regular_representation(
            parallel_transport_angles
        )

        print("partranspmatr.shape", parallel_transport_matrices.shape)
        theta = torch.tensor(2 * np.pi / N)
        rot_mat_3d = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta), 0],
                [torch.sin(theta), torch.cos(theta), 0],
                [0, 0, 1],
            ]
        )

        rot_x = torch.einsum("ij,vj->vi", rot_mat_3d, x)

        input = l2rBlock(x)
        rot_input = l2rBlock(rot_x)

        # A rotation of the reference frame also requires a rotation of the relative positions!
        rot_rel_pos_u = torch.einsum("ij,vnj->vni", rot_mat_3d[:2, :2], rel_pos_u)

        sa = GESelfAttentionBlock(N, in_channels=channels)

        output = sa(input, neighbors, mask, parallel_transport_matrices, rel_pos_u)
        rot_output = sa(
            rot_input, neighbors, mask, parallel_transport_matrices, rot_rel_pos_u
        )

        print(output[0])
        print(rot_output[0])

    def show_pooling():
        group_pool = GEGroupPooling(in_channels=3)
        input = torch.tensor(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
                [[25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36]],
            ],
            dtype=torch.float32,
        )
        print(out := group_pool(input))
        print(out.shape)

        global_pool = GEGlobalAveragePooling(in_channels=3)
        print(out := global_pool(out))
        print(out.shape)

    def check_gauge_invariance(data, angles, N, channels, verbose=True):
        # This tests wether the network (local2reg linear block, self attention block, group pool, global average pool)
        # is gauge invariant, performing a different rotation for each vertex

        x = data["features"]  # (N_v, 3)
        neighbors = data["neighbors"]  # (N_v, Max_Neighbors)
        parallel_transport_angles = data["g_qp"]  # (N_v, Max_Neighbors)

        rel_pos_u = data["u_q"]  # (N_v, Max_Neighbors, 2)
        mask = data["mask"]  # (N_v, Max_Neighbors)

        l2rBlock = GELocalToRegularLinearBlock(N, channels=channels)

        r2r = GEUtils.RegularToRegular(N)
        parallel_transport_matrices = r2r.extended_regular_representation(
            parallel_transport_angles
        )

        sa = GESelfAttentionBlock(N, in_channels=channels)

        # The parallel transport angles transform as: new_theta_nv = theta_nv + random_angle_v - random_angle_n
        new_parallel_transport_angles = (
            parallel_transport_angles + angles.unsqueeze(-1) - angles[neighbors]
        )

        rot_parallel_transport_matrices = r2r.extended_regular_representation(
            new_parallel_transport_angles
        )

        cos = torch.cos(angles)  # (N_v,)
        sin = torch.sin(angles)  # (N_v,)

        # fmt: off
        rot_mat_3d = torch.stack([
            torch.stack([cos, -sin, torch.zeros_like(cos)], dim=-1),
            torch.stack([sin,  cos, torch.zeros_like(cos)], dim=-1),
            torch.stack([torch.zeros_like(cos), torch.zeros_like(cos), torch.ones_like(cos)], dim=-1)
        ], dim=-2)
        # fmt: on

        x_rot = torch.einsum("vij,vj->vi", rot_mat_3d, x)
        rot_rel_pos_u = torch.einsum("vij,vnj->vni", rot_mat_3d[:, :2, :2], rel_pos_u)

        input = l2rBlock(x)
        rot_input = l2rBlock(x_rot)

        output = sa(input, neighbors, mask, parallel_transport_matrices, rel_pos_u)
        rot_output = sa(
            rot_input, neighbors, mask, rot_parallel_transport_matrices, rot_rel_pos_u
        )

        # Now let's apply group pooling and average pooling:
        group_poool = GEGroupPooling(in_channels=channels)
        global_pool = GEGlobalAveragePooling(in_channels=channels)

        out = global_pool(group_poool(output))
        rot_out = global_pool(group_poool(rot_output))

        if verbose:
            print(
                "gauge rotation at [0] in terms of 2pi/N: ",
                angles[0] / (2 * np.pi / N),
            )
            print("Original output of self attention block, [0]: \n", output[0])
            print(
                "New output of self attention block with changed gauge, [0]: \n",
                rot_output[0],
            )
            print("out: ", out)
            print("rot_out: ", rot_out)

        return out, rot_out

    def mean_gauge_violation(data, N, channels, trials):
        N_v = data["features"].shape[0]
        gauge_violation = 0
        for i in tqdm(range(trials)):
            angles = torch.randn((N_v,), dtype=torch.float32) * 2 * np.pi / N
            rot, rot_out = check_gauge_invariance(
                data, angles, N, channels, verbose=False
            )
            gauge_violation += torch.norm(rot - rot_out) / (
                (torch.norm(rot) + torch.norm(rot_out)) / 2
            )
        return gauge_violation / trials

    path = "../data/processed/T3.pt"
    data = torch.load(path, map_location="cpu")
    x = data["features"]  # (N_v, 3)
    neighbors = data["neighbors"]  # (N_v, Max_Neighbors)
    parallel_transport_angles = data["g_qp"]  # (N_v, Max_Neighbors)
    rel_pos_u = data["u_q"]  # (N_v, Max_Neighbors, 2)
    mask = data["mask"]  # (N_v, Max_Neighbors)

    N = 9
    channels = 12

    l2rBlock = GELocalToRegularLinearBlock(N, channels=channels)

    r2r = GEUtils.RegularToRegular(N)
    parallel_transport_matrices = r2r.extended_regular_representation(
        parallel_transport_angles
    )

    sa = GESelfAttentionBlock(N, in_channels=channels)

    out = sa(l2rBlock(x), neighbors, mask, parallel_transport_matrices, rel_pos_u)
