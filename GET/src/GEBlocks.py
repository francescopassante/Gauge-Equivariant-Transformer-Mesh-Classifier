import GEUtils
import numpy as np
import torch
import torch.nn as nn


class LocalToRegularLinearBlock(nn.Module):
    """
    A linear layer that implements equivariance from the rho_local representation to the regular representation for the cyclic group C_N.
    """

    def __init__(self, N, num_fields):
        """
        Args:
            N: Dimension of the regular representation (C_N).
            num_fields: Number of regular output fields.
        """
        super().__init__()
        self.N = N
        utils = GEUtils.LocalToRegular(N)
        W_basis = utils.local_to_regular_basis()
        self.register_buffer(
            "basis", torch.stack(W_basis)
        )  # This registers the basis as a non-learnable buffer
        self.num_basis = self.basis.shape[0]  # Number of basis matrices
        self.num_fields = num_fields

        # Learnable coefficients for each basis matrix for each output field
        # Initializing with small random values
        self.weights = nn.Parameter(torch.randn(num_fields, self.num_basis) * 0.02)

    def forward(self, x):
        """
        Args:
            x: Input features (rho_local) of shape (Batch, Num_Points, 3)
        Returns:
            Output feature fields of shape (Batch, Num_Points, num_fields * N)
        """
        # 1. Compute the kernel for each field: W = sum(a_i * W_basis_i)
        # Resulting shape: (num_fields, N, 3)
        combined_kernels = torch.einsum("fk,knm->fnm", self.weights, self.basis)

        # 2. Reshape kernels to a single large weight matrix for efficient computation
        # Shape: (num_fields, N, 3) -> (num_fields * N, 3)
        W_final = combined_kernels.view(self.num_fields * self.N, 3)

        # 3. Apply the linear transformation to the input features
        # (B, P, 3) @ (3, num_fields*N) -> (B, P, num_fields*N)
        out = torch.matmul(x, W_final.t())

        return out


class SelfAttentionBlock(nn.Module):
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
        self.register_buffer("value_basis_zero_order", value_basis[0])
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
        N_v, _ = x.shape

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
        print("f'_q shape: ", f_prime_q.shape)

        # 2. Compute Attention Scores
        # print("x shape: ", x.shape)
        K = self.W_K(x)  # .view(N_v, -1, self.n_heads, self.d_k)
        print("K shape: ", K.shape)
        Q = self.W_Q(f_prime_q)
        print("Q shape: ", Q.shape)

        score = (
            torch.relu(Q + K.unsqueeze(1)).mean(dim=-1).masked_fill(~mask, 0)
        )  # [N_v, Max_Neigh, in_channels]

        print("Score shape: ", score.shape)
        score_denominator = score.sum(dim=-1).clamp(min=1e-8)

        attention = score / score_denominator.unsqueeze(-1)
        print("Attention shape: ", attention.shape)

        # 3. Compute Values using Equivariant Kernel W_V(u)
        # W_V(u) = W0 + W1*u1 + W2*u2 ... (Taylor Expansion)

        print("rel_pos_u shape: ", rel_pos_u.shape)
        u_0 = rel_pos_u[..., 0]
        u_1 = rel_pos_u[..., 1]
        u_0_squared = u_0**2
        u_1_squared = u_1**2
        u_0_u_1 = (
            2 * u_0 * u_1
        )  # This 2 factor i think is fundamental, goes back to the SVD solution and form of F for the second order

        zero_order = torch.einsum(
            "cb,boij->coij",
            self.value_matrix_zero_order_params,
            self.value_basis_zero_order,
        ).squeeze(1)  # [N, N]

        first_order = torch.einsum(
            "cb,boij,vno->vncij",
            self.value_matrix_first_order_params,
            self.value_basis_first_order,
            rel_pos_u,
        )

        second_order = torch.einsum(
            "cb,boij,vno->vncij",
            self.value_matrix_second_order_params,
            self.value_basis_second_order,
            torch.stack([u_0_squared, u_1_squared, u_0_u_1], dim=-1),
        )

        value_kernel = (
            zero_order.unsqueeze(0).unsqueeze(0) + first_order + second_order
        )  # [in_channels, N_v, Max_Neigh, N, N]

        # Apply value function to transported features
        # V = W_V(u) * f_prime_q
        f_prime_q = f_prime_q.view(N_v, -1, self.in_channels, self.N)

        values = torch.einsum("vncij,vncj->vnci", value_kernel, f_prime_q)

        # 4. Aggregation
        out = torch.einsum("vn,vnci->vci", attention, values)  # [N_v, in_channels, N]
        return out


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

        equivariant_layer = LocalToRegularLinearBlock(N=9, num_fields=12)

        # Rotate the input by 2pi/9 (the angle corresponding to the cyclic group C9) and check the output
        theta = torch.tensor(2 * np.pi / 9, dtype=torch.float32)

        input = torch.randn(1, 1, 3)  # Original input
        rotated_input = rotate_input(input, theta)

        output = equivariant_layer(input).view(1, 1, 12, 9)
        rotated_output = equivariant_layer(rotated_input).view(1, 1, 12, 9)

        print(output[0][0][3])  # Should be (1, 1, 12, 9)
        print(rotated_output[0][0][3])  # Should be (1, 1, 12, 9)

    # load the data
    path = "../data/processed/T3.pt"
    data = torch.load(path, map_location="cpu")
    x = data["features"]  # (N_v, 3)
    neighbors = data["neighbors"]  # (N_v, Max_Neighbors)
    parallel_transport_angles = data["g_qp"]  # (N_v, Max_Neighbors, N, N)
    rel_pos_u = data["u_q"]  # (N_v, Max_Neighbors, 2)
    mask = data["mask"]  # (N_v, Max_Neighbors)

    N = 3
    channels = 12

    input = LocalToRegularLinearBlock(N, num_fields=channels)(
        x
    )  # (N_v, in_channels * N)

    r2r = GEUtils.RegularToRegular(N)
    parallel_transport_matrices = r2r.extended_regular_representation(
        parallel_transport_angles
    )

    # print("input: ", input.shape)
    # print("neighbors: ", neighbors.shape)
    # print("parallel transport matrices: ", parallel_transport_matrices.shape)
    # print("relative positions: ", rel_pos_u.shape)

    sa = SelfAttentionBlock(N, in_channels=channels)

    output = sa(input, neighbors, mask, parallel_transport_matrices, rel_pos_u)
