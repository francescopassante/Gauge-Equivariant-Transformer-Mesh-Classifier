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
        self.weights = nn.Parameter(torch.empty(out_channels, self.num_basis))
        nn.init.kaiming_normal_(self.weights, nonlinearity="relu")

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


class GERegularToRegularLinearBlock(nn.Module):
    """
    A linear gauge-equivariant map between regular representations.
    """

    def __init__(self, N, in_channels, out_channels):
        super().__init__()
        self.N = N
        self.in_channels = in_channels
        self.out_channels = out_channels

        basis = GEUtils.RegularToRegular(N).regular_to_regular_basis()
        self.register_buffer("basis", torch.stack(basis))
        self.num_basis = self.basis.shape[0]

        # Learnable coefficients: [out_channels, in_channels, num_basis]
        self.weights = nn.Parameter(
            torch.empty(out_channels, in_channels, self.num_basis)
        )
        nn.init.kaiming_normal_(self.weights, nonlinearity="relu")

    def forward(self, x):
        # x is [N_v, in_channels, N]
        # Construct the transformation matrix W: [out_channels, in_channels, N, N]
        W = torch.einsum("ocb, bij -> ocij", self.weights, self.basis)

        # Apply W to x
        # Output is [N_v, out_channels, N]
        return torch.einsum("ocij, vcj -> voi", W, x)


class GESelfAttentionBlock(nn.Module):
    """
    Gauge equivariant self attention block with optional multiple heads. To achieve gauge equivariance, features are parallel transported
    from neighbors of each vertex to each vertex. The Key and Query maps are linear maps from (in_channels, N) to (N),
    built as: (in_channels, N) -> reg2reg linear map for each channel -> (in_channels, N) -> sum over all channels -> (N).
    The gauge invariant score is calculated as P(ReLU(K + Q)) where ReLU is computed element-wise, and P is the average
    over all N dimensions. The attention between vertex v and neighbor n is this score normalized for all neighbors of v.
    The Value map is built as W_V(u) applied to the feature vector f(u), parallel transported to p.

    This version supports multiple attention heads. Heads are simple and independent; their scores are used to weight
    the same values and the head outputs are averaged back into the original channel representation.
    """

    def __init__(self, N, in_channels, num_heads=1):
        super().__init__()
        self.N = N
        self.in_channels = in_channels
        self.num_heads = num_heads

        # Equivariant basis for Query and Key linear maps
        basis = GEUtils.RegularToRegular(N).regular_to_regular_basis()
        self.register_buffer("reg_to_reg_basis", torch.stack(basis))

        # Query and Key coefficients are [num_heads, in_channels, len_basis]
        self.query_coeffs = nn.Parameter(
            torch.empty(num_heads, in_channels, len(basis))
        )
        self.key_coeffs = nn.Parameter(torch.empty(num_heads, in_channels, len(basis)))
        nn.init.xavier_uniform_(self.query_coeffs)
        nn.init.xavier_uniform_(self.key_coeffs)

        # The value matrix is given by a second order Taylor expansion in the relative position u.
        value_basis = GEUtils.RegularToRegular(N).get_taylor_basis()
        self.register_buffer("value_basis_zero_order", value_basis[0].squeeze(1))
        self.register_buffer("value_basis_first_order", value_basis[1])
        self.register_buffer("value_basis_second_order", value_basis[2])

        # Value parameters: [H, in_channels, basis_dim]
        self.value_matrix_zero_order_params = nn.Parameter(
            torch.empty(num_heads, in_channels, self.value_basis_zero_order.shape[0])
        )
        self.value_matrix_first_order_params = nn.Parameter(
            torch.empty(num_heads, in_channels, self.value_basis_first_order.shape[0])
        )
        self.value_matrix_second_order_params = nn.Parameter(
            torch.empty(num_heads, in_channels, self.value_basis_second_order.shape[0])
        )

        nn.init.xavier_uniform_(self.value_matrix_zero_order_params)
        nn.init.xavier_uniform_(self.value_matrix_first_order_params)
        nn.init.xavier_uniform_(self.value_matrix_second_order_params)

        self.W_M = GERegularToRegularLinearBlock(
            N, in_channels * num_heads, in_channels
        )

    def W_Q(self, x):
        # W_Q is a NxN matrix for each head and each channel -> [H, in_channels, N, N]
        W_Q = torch.einsum("hcb, bij -> hcij", self.query_coeffs, self.reg_to_reg_basis)
        # x = [N_v, in_channels, N]
        # Sum over channels producing one N-vector per head: result [N_v, H, N]
        return torch.einsum("hcij, vcj -> vhi", W_Q, x)

    def W_K(self, fprime):
        # fprime is [N_v, MAX_NEIGH, in_channels, N]
        # W_K is [H, in_channels, N, N]
        W_K = torch.einsum("hcb, bij -> hcij", self.key_coeffs, self.reg_to_reg_basis)
        # Sum over channels -> [N_v, MAX_NEIGH, H, N]
        return torch.einsum("hcij, vncj -> vnhi", W_K, fprime)

    def forward(self, x, neighbors, mask, parallel_transport_matrices, rel_pos_u):
        """
        Args:
            x: [N_v, in_channels, N] - Center features
            neighbors: [N_v, MAX_NEIGH] - Indices of neighbors
            mask: [N_v, MAX_NEIGH] - Binary mask for valid neighbors
            parallel_transport_matrices: [N_v, MAX_NEIGH, N, N] - rho_tilde(theta)
            rel_pos_u: [N_v, MAX_NEIGH, 2] - Logarithmic map coordinates u_q
        """
        N_v = x.shape[0]

        # Gather neighbors and apply mask once (no in-place write on a large temporary)
        x_neigh = x[neighbors].view(N_v, -1, self.in_channels, self.N)
        x_neigh = x_neigh * mask.unsqueeze(-1).unsqueeze(-1).to(x_neigh.dtype)

        # Parallel transport features of neighbors to center vertices -> [N_v, MAX_NEIGH, in_channels, N]
        f_prime_q = torch.einsum(
            "vnij,vncj->vnci", parallel_transport_matrices, x_neigh
        )

        # Compute Query and Key
        Q = self.W_Q(x)  # [N_v, H, N]
        K = self.W_K(f_prime_q)  # [N_v, MAX_NEIGH, H, N]

        # Compute attention scores per head
        score = torch.relu(Q.unsqueeze(1) + K).mean(dim=-1)  # [N_v, MAX_NEIGH, H]
        score = score.masked_fill(~mask.unsqueeze(-1), 0.0)

        score_denominator = score.sum(dim=1)  # [N_v, H]
        # Dead-attention fallback: if all scores are zero for a vertex/head, use
        # uniform attention over valid neighbors instead of a near-zero denominator.
        dead_mask = score_denominator < 1e-6  # [N_v, H]
        n_valid = mask.sum(dim=1, keepdim=True).clamp(min=1).float()  # [N_v, 1]
        uniform = mask.unsqueeze(-1).float() / n_valid.unsqueeze(-1)  # [N_v, MAX_NEIGH, 1]
        attention = torch.where(
            dead_mask.unsqueeze(1),
            uniform,
            score / score_denominator.unsqueeze(1).clamp(min=1e-6),
        )  # [N_v, MAX_NEIGH, H]

        u_0 = rel_pos_u[..., 0]
        u_1 = rel_pos_u[..., 1]
        u_quad = torch.stack([u_0**2, 2 * u_0 * u_1, u_1**2], dim=-1)

        # Apply value function to transported features
        # W0: [H, in_channels, N, N]
        W0 = torch.einsum(
            "hcb, bij -> hcij",
            self.value_matrix_zero_order_params,
            self.value_basis_zero_order,
        )
        values = torch.einsum("hcij, vncj -> vnhci", W0, f_prime_q)

        # First order — slice over the positional dimension to avoid a large
        # [N_v, MAX_NEIGH, C, 2, N] intermediate tensor in memory.
        W1 = torch.einsum(
            "hcb, boij -> hcoij",
            self.value_matrix_first_order_params,
            self.value_basis_first_order,
        )
        for o_idx in range(rel_pos_u.shape[-1]):  # 2 iterations
            W1_o = W1[:, :, o_idx, :, :]  # [H, C, N, N]
            values = values + torch.einsum(
                "hcij, vncj, vn -> vnhci", W1_o, f_prime_q, rel_pos_u[..., o_idx]
            )

        # Second order — same trick over u_quad's 3 components.
        W2 = torch.einsum(
            "hcb, boij -> hcoij",
            self.value_matrix_second_order_params,
            self.value_basis_second_order,
        )
        for o_idx in range(u_quad.shape[-1]):  # 3 iterations
            W2_o = W2[:, :, o_idx, :, :]  # [H, C, N, N]
            values = values + torch.einsum(
                "hcij, vncj, vn -> vnhci", W2_o, f_prime_q, u_quad[..., o_idx]
            )

        # Aggregation across neighbors using per-head attention
        head_outputs = torch.einsum("vnh, vnhci -> vhci", attention, values)

        # Use reg to reg linear block to mix heads and reduce back to in_channels:
        out = self.W_M(
            head_outputs.view(N_v, self.num_heads * self.in_channels, self.N)
        )

        return out


class GEResNetBlock(nn.Module):
    """
    A single Gauge Equivariant ResNet Block containing two multi-head self attention layers and two NormLayers
    """

    def __init__(self, N, channels, heads):
        super().__init__()
        self.norm1 = GELayerNorm(channels)
        self.mhsa1 = GESelfAttentionBlock(N, channels, heads)

        self.norm2 = GELayerNorm(channels)
        self.mhsa2 = GESelfAttentionBlock(N, channels, heads)

    def forward(self, x, neighbors, mask, pt_matrices, rel_pos_u):
        # Pre-norm style: normalize before each attention layer, add residual after.
        out = x + self.mhsa1(self.norm1(x), neighbors, mask, pt_matrices, rel_pos_u)
        out = out + self.mhsa2(self.norm2(out), neighbors, mask, pt_matrices, rel_pos_u)
        return out


class GELayerNorm(nn.Module):
    """
    Gauge Equivariant Layer Normalization.
    Computes mean and variance over the channels and the N dimensions.
    """

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        # Affine parameters must be shared across the N dimension
        # to guarantee gauge equivariance.
        self.weight = nn.Parameter(torch.ones(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        # x shape: [N_v, channels, N]
        # Compute mean and variance over channels and N
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply affine transformation
        return x_norm * self.weight + self.bias


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

        global_pool = GEGlobalAveragePooling()
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
