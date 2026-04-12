import numpy as np
import torch


class RegularToRegular:
    def __init__(self, N):
        self.N = N
        self.A = self.get_dft_matrix()

    def regular_to_regular_basis(self):
        """
        Returns the basis of linear maps W_i that satisfy the equivariance condition:
        rho @ W_i = W_i @ rho
        where rho is the regular representation. In this case, since regular to regular, a basis is given by circulant matrices
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
        Computes the basis of linear maps for a value function Taylor expansion up to order 2,
        satisfying Eqn. (78) for the regular representation of C_N.
        """
        # 2. Regular Representation rho_reg(Theta0)
        # This is a cyclic shift matrix
        theta = 2 * np.pi / self.N
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # rho_reg = self.extended_regular_representation(theta)
        rho_reg = torch.zeros((self.N, self.N))
        for i in range(self.N):
            rho_reg[(i + 1) % self.N, i] = 1

        # 3. Solve per Taylor Order
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
        # Pre-compute parts of term1 to ensure contiguity and correctness
        # Eqn 78: rho_out(Theta0^-1) and rho_in(Theta0)^T
        # For permutation matrices, Inverse == Transpose
        rho_out_inv = rho_reg.T.contiguous()
        rho_in_T = rho_reg.T.contiguous()

        # This is the (rho_out_inv \otimes rho_in_T) part
        rho_kernel = torch.kron(rho_out_inv, rho_in_T).contiguous()

        for order in range(3):
            n_terms = order + 1
            F = Fs[order]

            # 4. Construct the Constraint Matrix M (Eqn 78)
            # M = I_n+1 \otimes (rho_out(inv) \otimes rho_in^T) - F \otimes I_N*N
            term1 = torch.kron(torch.eye(n_terms, dtype=torch.float64), rho_kernel)
            term2 = torch.kron(F, torch.eye(self.N * self.N, dtype=torch.float64))

            M = term1 - term2

            # 5. Solve via SVD
            U, S, Vh = torch.linalg.svd(M)

            # Extract null space (singular values ~ 0)
            tol = 1e-7
            null_mask = S < tol
            bases = Vh[null_mask]

            # Reshape to [n_bases, n_terms, N, N]
            if bases.shape[0] > 0:
                # Use reshape to avoid contiguous subspace errors
                all_bases.append(bases.reshape(-1, n_terms, self.N, self.N).float())

        return all_bases

    def get_dft_matrix(self):
        A = torch.zeros((self.N, self.N))
        A[:, 0] = 1.0 / np.sqrt(self.N)

        for k in range(1, (self.N // 2) + 1):
            for j in range(self.N):
                angle = (2 * np.pi * j * k) / self.N
                A[j, 2 * k - 1] = np.sqrt(2.0 / self.N) * np.cos(angle)
                A[j, 2 * k] = np.sqrt(2.0 / self.N) * np.sin(angle)
        return A

    def extended_regular_representation(self, theta):
        """
        theta: shape (V, K)  # vertices, neighbors

        returns:
            rho: shape (V, K, N, N)
        """
        V, K = theta.shape
        device = theta.device

        # Initialize D_theta batch
        D_theta = torch.zeros(V, K, self.N, self.N, device=device)

        # k = 0 block (scalar irrep)
        D_theta[..., 0, 0] = 1.0

        # Higher frequencies
        for k in range(1, (self.N // 2) + 1):
            cos_kt = torch.cos(k * theta)  # (V, K)
            sin_kt = torch.sin(k * theta)  # (V, K)

            i = 2 * k - 1
            j = 2 * k

            D_theta[..., i, i] = cos_kt
            D_theta[..., i, j] = -sin_kt
            D_theta[..., j, i] = sin_kt
            D_theta[..., j, j] = cos_kt

        # Apply change of basis: A @ D @ A^T in batch
        A = self.A  # (N, N)

        rho = torch.matmul(
            torch.matmul(A, D_theta),  # broadcasted
            A.T,
        )

        return rho.round(decimals=6)


class LocalToRegular:
    def __init__(self, N):
        self.N = N
        self.rho_in = self.get_local_representation_rho_in()
        self.rho_out = self.get_regular_representation_rho_out()

    def local_to_regular_basis(self):
        """
        Computes the basis of linear maps W_i that satisfy the equivariance condition:
        rho_regular @ W_i = W_i @ rho_local
        """

        # Based on Eqn. (78), for n=0: (rho_out_inv \otimes rho_in) - I
        I_in = np.eye(3)
        I_out = np.eye(self.N)
        M = np.kron(I_out, self.rho_in.T) - np.kron(self.rho_out, I_in)

        # 3. Solve via SVD
        u, s, vh = np.linalg.svd(M)

        # The basis vectors are the rows of vh corresponding to zero singular values
        tol = 1e-10
        basis_vectors = vh[s < tol]

        # 4. Reshape to get W_i matrices (N x 3)
        return [
            torch.tensor(v.reshape(self.N, 3), dtype=torch.float32)
            for v in basis_vectors
        ]

    def get_local_representation_rho_in(self):
        # Rappresentazione locale: rotazione di 2pi/N intorno all'asse z
        theta = 2 * np.pi / self.N
        return np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

    def get_regular_representation_rho_out(self):
        rho_out = np.zeros((self.N, self.N))
        for i in range(self.N):
            rho_out[(i + 1) % self.N, i] = 1
        return rho_out


if __name__ == "__main__":
    N = 3
    regular_to_regular = RegularToRegular(N)

    taylor_basis = regular_to_regular.get_taylor_basis()
    print(len(taylor_basis))
    print(taylor_basis[0].shape)
    print(taylor_basis[1].shape)
    print(taylor_basis[2].shape)

    # LEt's check the equivariance condition of a generic linear combination of the basis elements
    # 0 order order basis is 9 dimensional, 1st order basis is 18 dimensional, 2nd order basis is 27 dimensional

    # GEMINI:
    # 1. Fissa i coefficienti (Parametri del modello)
    coeff_0 = torch.randn(taylor_basis[0].shape[0])
    coeff_1 = torch.randn(taylor_basis[1].shape[0])
    coeff_2 = torch.randn(taylor_basis[2].shape[0])

    def value_matrix_fixed(c0, c1, c2, taylor_basis, u):
        N = taylor_basis[0].shape[-1]
        W = torch.zeros((N, N), dtype=torch.float32)

        # Ordine 0
        for i in range(taylor_basis[0].shape[0]):
            W += c0[i] * taylor_basis[0][0][0]
        # Ordine 1
        for i in range(taylor_basis[1].shape[0]):
            W += c1[i] * (taylor_basis[1][0][0] * u[0] + taylor_basis[1][0][1] * u[1])
        # # Ordine 2
        for i in range(taylor_basis[2].shape[0]):
            W += c2[i] * (
                taylor_basis[2][0][0] * u[0] ** 2
                + taylor_basis[2][0][1] * 2 * u[0] * u[1]
                + taylor_basis[2][0][2] * u[1] ** 2
            )
        return W

    # 2. Definisci la rotazione (deve essere la stessa usata in get_taylor_basis)
    theta = torch.tensor([[2 * np.pi / N]])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Matrice di rotazione passiva (cambio di coordinate)
    R_inv = torch.tensor([[cos_t, sin_t], [-sin_t, cos_t]], dtype=torch.float32)

    u = torch.randn(2)
    u_rotated = u @ R_inv.T  # Corrisponde a Theta_0^-1 * u

    # 3. Verifica l'equazione: rho(g) * W(Theta^-1 * u) = W(u) * rho(g)
    rho_tilde = regular_to_regular.extended_regular_representation(theta).float()
    print("rho_tilde: \n", rho_tilde)

    lhs = rho_tilde @ value_matrix_fixed(
        coeff_0, coeff_1, coeff_2, taylor_basis, u_rotated
    )
    rhs = value_matrix_fixed(coeff_0, coeff_1, coeff_2, taylor_basis, u) @ rho_tilde

    print("LHS:\n", lhs)
    print("RHS:\n", rhs)
