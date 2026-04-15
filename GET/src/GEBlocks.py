import GEUtils
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class GELocalToRegularLinearBlock(nn.Module):
    """
    A layer that implements a linear gauge equivariant map from the rho_local representation to the regular representation C_N
    """

    def __init__(self, N, out_channels):
        """
        Args:
            N: Dimension of the regular representation (C_N).
            out_channels: Number of regular output fields.
        """
        super().__init__()
        self.N = N
        utils = GEUtils.LocalToRegular(N)

        # Basis of the equivariant kernel
        W_basis = utils.local_to_regular_basis()
        self.register_buffer(
            "basis", torch.stack(W_basis)
        )  # This registers the basis as a non-learnable buffer
        self.num_basis = self.basis.shape[0]  # Number of basis matrices
        self.out_channels = out_channels

        # Learnable coefficients for each basis matrix for each output field
        # Initializing with small random values
        self.weights = nn.Parameter(torch.randn(out_channels, self.num_basis) * 0.02)

    def forward(self, x):
        """
        Args:
            x: Input features (rho_local) of shape (Num_Points, 3)
        Returns:
            Output feature fields of shape (Num_Points, channels, N)
        """
        # Compute the kernel for each channel: W = sum(a_i * W_basis_i)
        # Resulting shape: (channels, N, 3)
        combined_kernels = torch.einsum("cb,bij->cij", self.weights, self.basis)

        # Reshape kernels to a single large weight matrix for efficient computation
        # Shape: (channels, N, 3) -> (channels * N, 3)
        W_final = combined_kernels.view(self.out_channels * self.N, 3)

        # 3. Apply the linear transformation to the input features
        out = torch.matmul(x, W_final.t()).view(x.shape[0], self.out_channels, self.N)

        return out


class GESelfAttentionBlock(nn.Module):
    """
    Gauge equivariant self attention block. To achieve gauge equivariance, features are parallel transported
    from neighbors of each vertex to each vertex. The Key and Query maps are linear maps from (in_channels, N) to (N),
    built as: (in_channels, N) -> reg2reg linear map for each channel -> (in_channels, N) -> sum over all channels -> (N).
    The gauge invariant score is calculated as P(ReLU(K + Q)) where ReLU is computed element-wise, and P is the average
    over all N dimensions. The attention between vertex v and neighbor n is just this score normalized for all neighbors of v.
    The Value map is built as W_V(u) applied to the feature vector f(u), parallel transported to p.
    """

    def __init__(self, N, in_channels):
        super().__init__()
        self.N = N

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
        )  # Reshaping because at zero order there is only one matrix (W_0), useful for later calculations.
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

    def W_Q(self, x):
        # W_Q is a NxN matrix for each channel -> [in_channels, N, N]
        W_Q = torch.einsum("cb, bij -> cij", self.query_coeffs, self.reg_to_reg_basis)
        x = x.view(x.shape[0], self.in_channels, self.N)  # x = [N_v, in_channels, N]
        # Sum over channels (the most general equivariant map is the sum of the most general maps for each individual channel)
        return torch.einsum("cij, vcj -> vi", W_Q, x)

    def W_K(self, fprime):
        # fprime is [N_v, MAX_NEIGH, in_channels, N]
        # W_K is [in_channels, N, N]
        W_K = torch.einsum("cb, bij -> cij", self.key_coeffs, self.reg_to_reg_basis)
        # Sum over channels
        return torch.einsum("cij, vncj -> vni", W_K, fprime)

    def forward(self, x, neighbors, mask, parallel_transport_matrices, rel_pos_u):
        """
        Args:
            x: [N_v, in_channels * N] - Center features
            neighbors: [N_v, MAX_NEIGH] - Indices of neighbors
            mask: [N_v, MAX_NEIGH] - Binary mask for valid neighbors
            parallel_transport_matrices: [N_v, MAX_NEIGH, N, N] - rho_tilde(theta)
            rel_pos_u: [N_v, MAX_NEIGH, 2] - Logarithmic map coordinates u_q
        """
        N_v, chan, n = x.shape

        # x_neigh : [N_v, MAX_NEIGH, channels, N]
        x_neigh = x[neighbors]
        x_neigh = x_neigh.view(N_v, -1, self.in_channels, self.N)

        # Zero-out "fake neighbors" so that x_neigh[v][n] is zero if n > actual number of neighbors for vertex v
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # [N_v, Max_N, 1, 1]
        x_neigh = x_neigh * mask_expanded

        # Parallel transport features of neighbors to center vertices
        f_prime_q = torch.einsum(
            "vnij,vncj->vnci", parallel_transport_matrices, x_neigh
        )

        # Compute Query and Key
        Q = self.W_Q(x)
        K = self.W_K(f_prime_q)

        # Compute attention
        score = torch.relu(Q.unsqueeze(1) + K).mean(dim=-1).masked_fill(~mask, 0)

        score_denominator = score.sum(dim=-1).clamp(min=1e-8)
        attention = score / score_denominator.unsqueeze(-1)

        # Compute Values using Equivariant Kernel W_V(u)
        # W_V(u) = W0 + W1*u1 + W2*u2 +W3u1^2 + 2*W4 u1u2 + W5 u2^2  (Taylor Expansion)

        u_0 = rel_pos_u[..., 0]
        u_1 = rel_pos_u[..., 1]
        u_0_squared = u_0**2
        u_1_squared = u_1**2
        u_0_u_1 = (
            2 * u_0 * u_1
        )  # This 2 factor is necessary because of how we defined F for the second order kernel

        # Apply value function to transported features
        # V = W_V(u) * f'(u)

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

        W1_local = torch.einsum("coij, vno -> vncij", W1, rel_pos_u)  # [V, N, C, I, J]
        values += torch.einsum("vncij, vncj -> vnci", W1_local, f_prime_q)

        del W0
        del W1  # Forza la liberazione della memoria, visto che non servono
        del W1_local

        W2 = torch.einsum(
            "cb, boij -> coij",
            self.value_matrix_second_order_params,
            self.value_basis_second_order,
        )

        u_quad = torch.stack([u_0_squared, u_0_u_1, u_1_squared], dim=-1)

        W2_local = torch.einsum("coij, vno -> vncij", W2, u_quad)  # [V, N, C, I, J]
        values += torch.einsum("vncij, vncj -> vnci", W2_local, f_prime_q)
        del W2
        del W2_local  # Forza la liberazione

        # Aggregation
        out = torch.einsum("vn,vnci->vci", attention, values)  # [N_v, in_channels, N]
        return out


class GEGroupPooling(nn.Module):
    """
    Group pooling takes feature field [N_v, in_channels, N] and takes the maximum over each channel -> [N_v, in_channels].
    This operation is gauge invariant.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.max(dim=-1)[0]  # [N_v, in_channels]


class GEGlobalAveragePooling(nn.Module):
    """
    GlobalAveragePooling averages the feature field [N_v, in_channels] over the whole mesh, output is a single [in_channels] vector
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=0)  # [in_channels]


if __name__ == "__main__":

    def check_equivariance_l2r(N, out_channels, k):
        """
        Demo to show that the local2regular layers is equivariant under a global rotation:
        if i rotate the local feature by an angle 2kpi/N
        the resulting output will be cyclically shifted by k steps.
        """

        def rotate_input(x, theta):
            # Rotate around the z-axis by angle theta
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rotation_matrix = torch.tensor(
                [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]],
                dtype=torch.float32,
            )
            return x @ rotation_matrix.t()

        equivariant_layer = GELocalToRegularLinearBlock(N=N, out_channels=out_channels)

        # Rotate the input by 2pi/N (the angle corresponding to the cyclic group C9) and check the output
        theta = torch.tensor(2 * np.pi * k / N, dtype=torch.float32)

        input = torch.randn(1, 1, 3)  # Original input
        rotated_input = rotate_input(input, theta)

        output = equivariant_layer(input).view(1, 1, out_channels, N)
        rotated_output = equivariant_layer(rotated_input).view(1, 1, out_channels, N)

        print(output[0][0][3])
        print(rotated_output[0][0][3])

    def check_equivariance_sa(N, channels):
        """
        Demo to check equivariance under global rotation of the self attention block.
        If the local input is rotated by 2kpi/N, the regular field will be shifted.
        If the self attention block is equivariant, the output of the self attention block will be cyclically shifted aswell.
        """
        # Take random mesh
        path = "../data/processed/T3.pt"
        data = torch.load(path)
        x = data["features"]  # (N_v, 3)
        neighbors = data["neighbors"]  # (N_v, MAX_NEIGH)
        parallel_transport_angles = data["g_qp"]  # (N_v, MAX_NEIGH, N, N)
        rel_pos_u = data["u_q"]  # (N_v, MAX_NEIGH, 2)
        mask = data["mask"]  # (N_v, MAX_NEIGH)

        # local2regular block
        l2rBlock = GELocalToRegularLinearBlock(N=N, out_channels=channels)

        # Generate parallel transport matrices from parallel transport angles.
        r2r = GEUtils.RegularToRegular(N)
        parallel_transport_matrices = r2r.extended_regular_representation(
            parallel_transport_angles
        )

        # Generate rotation matrix of the local features
        theta = torch.tensor(2 * np.pi / N)
        rot_mat_3d = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta), 0],
                [torch.sin(theta), torch.cos(theta), 0],
                [0, 0, 1],
            ]
        )

        # Rotate local features
        rot_x = torch.einsum("ij,vj->vi", rot_mat_3d, x)

        # Compute output of the local2regular block
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
        """
        Demo to show mechanism of group pooling and global average pooling
        """
        group_pool = GEGroupPooling()
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
        """
        Demo to show gauge invariance of the (local2regular -> self attention -> group pool -> global average pool) pipeline
        Performs a different rotation of reference frame at each vertex -> needs to update relative positions and parallel transport angles
        """

        x = data["features"]
        neighbors = data["neighbors"]
        parallel_transport_angles = data["g_qp"]

        rel_pos_u = data["u_q"]
        mask = data["mask"]

        l2rBlock = GELocalToRegularLinearBlock(N, out_channels=channels)
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

        cos = torch.cos(angles)
        sin = torch.sin(angles)

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
        group_poool = GEGroupPooling()
        global_pool = GEGlobalAveragePooling()

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
        """
        Performs a general gauge transformation (angles not multiples of 2pi/N), and computes the difference in the output.
        Takes the average over <trials> runs.
        """
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
    neighbors = data["neighbors"]  # (N_v, MAX_NEIGH)
    parallel_transport_angles = data["g_qp"]  # (N_v, MAX_NEIGH)
    rel_pos_u = data["u_q"]  # (N_v, MAX_NEIGH, 2)
    mask = data["mask"]  # (N_v, MAX_NEIGH)

    N = 9
    channels = 12

    l2rBlock = GELocalToRegularLinearBlock(N, channels=channels)

    r2r = GEUtils.RegularToRegular(N)
    parallel_transport_matrices = r2r.extended_regular_representation(
        parallel_transport_angles
    )

    sa = GESelfAttentionBlock(N, in_channels=channels)

    out = sa(l2rBlock(x), neighbors, mask, parallel_transport_matrices, rel_pos_u)
