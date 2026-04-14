import numpy as np
import torch


class RegularToRegular:
    """
    A class of utilities to handle mappings from regular fields to regular fields.
    Useful in Query, Key, Value maps in the self attention block
    """

    def __init__(self, N):
        self.N = N
        self.A = self.get_dft_matrix()

    def regular_to_regular_basis(self):
        """
        Returns the basis of linear maps W_i that satisfy the equivariance condition:
        rho @ W_i = W_i @ rho
        where rho is the regular representation. In this case, since regular to regular,
        a basis is given by all (N) NxN circulant matrices
        """
        basis = []
        for i in range(self.N):
            W_i = np.zeros((self.N, self.N))
            for j in range(self.N):
                W_i[j, (j + i) % self.N] = 1.0
            basis.append(W_i)

        return [torch.tensor(v, dtype=torch.float32) for v in basis]

    def get_taylor_basis(self):
        """
        Taylor expanding the value function W_N_v(u) up to second order in u and imposing equivariance
        one finds a linear equation for order W_0, a coupled linear equation for [W_1, W_2] and a coupled
        linear equation for [W_3, W_4, W_5]. This function computes all bases for the zero, first and second order
        via SN_vD. See eq. (78) of GET paper for more info.
        """

        # Regular Representation rho_reg(Theta0)
        # This is a cyclic shift matrix
        theta = 2 * np.pi / self.N
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        rho_reg = torch.zeros((self.N, self.N))
        for i in range(self.N):
            rho_reg[(i + 1) % self.N, i] = 1

        # We solve the equation (78) order by order, Fs contains the various F for each order
        Fs = [
            torch.tensor([[1.0]], dtype=torch.float64),
            torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], dtype=torch.float64),
            torch.tensor(
                [
                    [cos_t**2, -2 * cos_t * sin_t, sin_t**2],
                    [cos_t * sin_t, cos_t**2 - sin_t**2, -cos_t * sin_t],
                    [sin_t**2, 2 * cos_t * sin_t, cos_t**2],
                ],
                dtype=torch.float64,
            ),
        ]

        all_bases = []

        rho_out_inv = rho_reg.T.contiguous()
        rho_in_T = rho_reg.T.contiguous()

        rho_kernel = torch.kron(rho_out_inv, rho_in_T).contiguous()

        for order in range(3):
            n_terms = order + 1
            F = Fs[order]

            # Construct the Constraint Matrix
            term1 = torch.kron(torch.eye(n_terms, dtype=torch.float64), rho_kernel)
            term2 = torch.kron(F, torch.eye(self.N * self.N, dtype=torch.float64))

            M = term1 - term2

            # 5. Solve via SVD
            U, S, Vh = torch.linalg.svd(M)

            # Extract null space (singular values ~ 0)
            tol = 1e-7
            null_mask = S < tol
            bases = Vh[null_mask]

            all_bases.append(bases.reshape(-1, n_terms, self.N, self.N).float())

        return all_bases

    def get_dft_matrix(self):
        """Computes the real DFT matrix. Useful for finding the extended regular representation.
        The DFT basis is the one that diagonalizes the regular representation into its irreps"""
        A = torch.zeros((self.N, self.N), dtype=torch.float32)
        A[:, 0] = 1.0 / np.sqrt(self.N)

        for k in range(1, (self.N // 2) + 1):
            for j in range(self.N):
                angle = (2 * np.pi * j * k) / self.N
                A[j, 2 * k - 1] = np.sqrt(2.0 / self.N) * np.cos(angle)
                A[j, 2 * k] = np.sqrt(2.0 / self.N) * np.sin(angle)
        return A

    def extended_regular_representation(self, theta):
        """
        Computes the extended regular representation matrix corresponding to rotation angles theta: [N_v, MAX_NEIGH]
        """
        N_v, NEIGH = theta.shape
        device = theta.device
        N = self.N

        # Initialization
        D_theta = torch.zeros(N_v, NEIGH, N, N, device=device, dtype=torch.float32)

        # Scalar irrep
        D_theta[..., 0, 0] = 1.0
        # Higher frequencies irreps (for odd N, these are all 2x2 rotation matrices with higher and higher frequencies)
        for k in range(1, (N // 2) + 1):
            cos_kt = torch.cos(k * theta)
            sin_kt = torch.sin(k * theta)
            i, j = 2 * k - 1, 2 * k
            D_theta[..., i, i] = cos_kt
            D_theta[..., i, j] = -sin_kt
            D_theta[..., j, i] = sin_kt
            D_theta[..., j, j] = cos_kt

        A = self.A.to(device)

        # Implement basis change to obtain the actual rotation matrix
        rho = torch.matmul(
            torch.matmul(A, D_theta),  # broadcasted
            A.T,
        )

        # Rounding to avoid small numerical instabilities
        return rho.round(decimals=6)


class LocalToRegular:
    """
    A class of utilities to handle mappings from local fields to regular fields.
    Useful in the first mapping of the GET
    """

    def __init__(self, N):
        self.N = N
        self.rho_in = self.get_local_representation_rho_in()
        self.rho_out = self.get_regular_representation_rho_out()

    def local_to_regular_basis(self):
        """
        Computes the basis of linear maps W_i that satisfy the equivariance condition:
        rho_regular @ W_i = W_i @ rho_local
        """

        # Same logic as for taylor basis, indeed this is the same as zero-order taylor basis:
        I_in = np.eye(3)
        I_out = np.eye(self.N)
        M = np.kron(I_out, self.rho_in.T) - np.kron(self.rho_out, I_in)

        # Solve via SVD
        u, s, vh = np.linalg.svd(M)

        tol = 1e-10
        basis_vectors = vh[s < tol]

        return [
            torch.tensor(v.reshape(self.N, 3), dtype=torch.float32)
            for v in basis_vectors
        ]

    def get_local_representation_rho_in(self):
        """Local representation of the rotation group by theta = 2pi/N, just a 2d rotation matrix with the same theta"""
        theta = 2 * np.pi / self.N
        return np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

    def get_regular_representation_rho_out(self):
        """Regular representation of the rotation group by theta = 2pi/N, just a N-dimensional cyclical shift matrix"""
        rho_out = np.zeros((self.N, self.N))
        for i in range(self.N):
            rho_out[(i + 1) % self.N, i] = 1
        return rho_out


if __name__ == "__main__":
    """
    Small demo to show equivariance of Eq.(78): rho_regular(g) * W(Theta^-1 * u) = W(u) * rho_local(g)
    Where W is a generic linear combination of the zero order, first order and second order talyor basis (with proper u factors)
    """

    N = 3
    regular_to_regular = RegularToRegular(N)
    taylor_basis = regular_to_regular.get_taylor_basis()

    coeff_0 = torch.randn(taylor_basis[0].shape[0])
    coeff_1 = torch.randn(taylor_basis[1].shape[0])
    coeff_2 = torch.randn(taylor_basis[2].shape[0])

    def value_matrix_fixed(N, c0, c1, c2, taylor_basis, u):
        W = torch.zeros((N, N), dtype=torch.float32)

        # Zero order
        for i in range(taylor_basis[0].shape[0]):
            W += c0[i] * taylor_basis[0][0][0]
        # First order
        for i in range(taylor_basis[1].shape[0]):
            W += c1[i] * (taylor_basis[1][0][0] * u[0] + taylor_basis[1][0][1] * u[1])
        # Second order
        for i in range(taylor_basis[2].shape[0]):
            W += c2[i] * (
                taylor_basis[2][0][0] * u[0] ** 2
                + taylor_basis[2][0][1] * 2 * u[0] * u[1]
                + taylor_basis[2][0][2] * u[1] ** 2
            )
        return W

    theta = torch.tensor([[2 * np.pi / N]])
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # R is the basis rotation matrix (angle -theta), coordinates and local features will rotate the opposite way (+theta)
    R = torch.tensor([[cos_t, sin_t], [-sin_t, cos_t]], dtype=torch.float32)
    print("R: ", R)

    u = torch.randn(2)
    # u is rotated by +theta
    u_rotated = u @ R.T

    # rho_tilde corresponds to a rotation of +theta for a regular field
    rho_tilde = regular_to_regular.extended_regular_representation(theta).float()

    lhs = rho_tilde @ value_matrix_fixed(
        N, coeff_0, coeff_1, coeff_2, taylor_basis, u_rotated
    )
    rhs = value_matrix_fixed(N, coeff_0, coeff_1, coeff_2, taylor_basis, u) @ rho_tilde

    print("LHS:\n", lhs)
    print("RHS:\n", rhs)
