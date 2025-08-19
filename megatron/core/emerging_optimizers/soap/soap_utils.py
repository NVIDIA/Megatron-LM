from typing import List, Optional, Tuple
import torch

from megatron.core.emerging_optimizers import utils

__all__ = [
    "get_eigenbasis_eigh",
    "get_eigenbasis_qr",
]


def get_eigenbasis_eigh(
    kronecker_factor_list: List[torch.Tensor],
    eps: float = 1e-30,
    convert_to_float: bool = True,
) -> List[torch.Tensor]:
    """Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.

    Args:
        kronecker_factor_list: Matrix List to compute eigenbases of
        eps: Small offset to allow eigendecomposition with numerical stability
        convert_to_float: If True, preconditioner matrices and their corresponding
            orthonormal matrices will be cast to float. Otherwise, they are left in
            their original type.

    Returns:
        List[torch.Tensor]: List of orthonormal kronecker factor eigenbases matrices

    Example:
        .. code-block:: python

            # Create sample Kronecker factors (symmetric positive definite matrices)
            k_factor1 = torch.randn(4, 4)
            k_factor1 = k_factor1 @ k_factor1.T  # Make symmetric positive definite
            k_factor2 = torch.randn(5, 5)
            k_factor2 = k_factor2 @ k_factor2.T  # Make symmetric positive definite

            # Get orthogonal matrices for these factors
            ortho_matrices = get_eigenbasis_eigh([k_factor1, k_factor2])
            # ortho_matrices[0] has shape [4, 4] and ortho_matrices[1] has shape [5, 5]
    """
    casted_matrix_list: List[torch.Tensor] = []
    for kronecker_factor in kronecker_factor_list:
        if kronecker_factor.numel() == 0:
            casted_matrix_list.append(torch.empty(0, device=kronecker_factor.device))
            continue
        if convert_to_float:
            casted_matrix_list.append(kronecker_factor.to(torch.float))
        else:
            casted_matrix_list.append(kronecker_factor)

    eigenbasis_list: List[torch.Tensor] = []
    for kronecker_factor in casted_matrix_list:
        if kronecker_factor.numel() == 0:
            eigenbasis_list.append(torch.empty(0, device=kronecker_factor.device))
            continue
        try:
            # Create an identity matrix of the same size as m and add a small offset for stability.
            eye = torch.eye(
                kronecker_factor.shape[0],
                device=kronecker_factor.device,
                dtype=kronecker_factor.dtype if not convert_to_float else torch.float,
            )
            _, Q = torch.linalg.eigh(kronecker_factor + eps * eye)
        except Exception as e:
            print("Falling back to fp64 precision:", e)
            # Fallback to higher precision if the default precision fails
            m_fp64 = kronecker_factor.to(torch.float64)
            eye_fp64 = torch.eye(
                kronecker_factor.shape[0],
                device=kronecker_factor.device,
                dtype=torch.float64,
            )
            _, Q = torch.linalg.eigh(m_fp64 + eps * eye_fp64)
            # Convert the eigenbases back to the original dtype.
            Q = Q.to(kronecker_factor.dtype if not convert_to_float else torch.float)

        # Reverse the order of the eigenbases since torch.linalg.eigh returns them in ascending order
        Q = torch.flip(Q, [1])

        eigenbasis_list.append(Q)

    return eigenbasis_list


def get_eigenbasis_qr(
    kronecker_factor_list: List[torch.Tensor],
    eigenbasis_list: List[torch.Tensor],
    exp_avg_sq: torch.Tensor,
    convert_to_float: bool = True,
    use_adaptive_criteria: bool = False,
    adaptive_update_criteria_tolerance: Optional[float] = None,
    power_iter_steps: int = 1,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Updates the eigenbases of the preconditioner using power iteration and QR.

    Computes using multiple rounds of power iteration followed by QR decomposition (orthogonal iteration).

    Args:
        kronecker_factor_list: List containing preconditioner (GGT and GTG)
        eigenbasis_list: List containing eigenbases (QL and QR)
        exp_avg_sq: inner adam second moment (exp_avg_sq). This tensor is modified in-place.
        convert_to_float: If True, preconditioner matrices and their corresponding
            orthonormal matrices will be cast to float. Otherwise, they are left in
            their original type.
        use_adaptive_criteria: Whether to use update criteria strategy
        adaptive_update_criteria_tolerance: Tolerance threshold for the normalized off diagonal component of estimated eigenvalue matrix.
            If None, defaults to 1e-30, which means adaptive update criteria will be used whenever there is a small change in the estimated eigenvalues
            matrix and QR will be used.
        power_iter_steps: Number of power iteration steps to perform before QR decomposition.
            More steps can lead to better convergence but increased computation time.

    Returns:
        List[torch.Tensor]: Updated list of orthonormal kronecker factor eigenbases matrices
        torch.Tensor: Updated (sorted) inner adam second moment

    Example:
        .. code-block:: python

            # Create sample Kronecker factors (symmetric positive definite matrices)
            n, m = 10, 20
            k_factor1 = torch.randn(n, n)
            k_factor1 = k_factor1 @ k_factor1.T  # Make symmetric positive definite
            k_factor2 = torch.randn(m, m)
            k_factor2 = k_factor2 @ k_factor2.T  # Make symmetric positive definite

            # Get orthogonal matrices for these kronecker factors
            kronecker_factor_list = [k_factor1, k_factor2]
            eigenbasis_list = get_eigenbasis_eigh(kronecker_factor_list)

            # Perturb the kronecker factor matrices, simulating the effect of gradient updates
            perturbation = 1e-2*torch.randn(n, m)
            perturbed_kronecker_factor_list = [None, None]
            perturbed_kronecker_factor_list[0] = k_factor1 + perturbation@perturbation.T
            perturbed_kronecker_factor_list[1] = k_factor2 + perturbation.T@perturbation

            # Initialize exp_avg_sq tensor
            exp_avg_sq = torch.randn(n, m).abs()

            # Refine the orthogonal matrices using QR
            updated_ortho_matrices, updated_exp_avg_sq = get_eigenbasis_qr(
                perturbed_kronecker_factor_list,
                eigenbasis_list,
                exp_avg_sq
            )
    """
    if adaptive_update_criteria_tolerance is None:
        adaptive_update_criteria_tolerance = 1e-30

    casted_matrix_list: List[torch.Tensor] = []
    casted_eigenbasis_list: List[torch.Tensor] = []
    for kronecker_factor, eigenbasis in zip(kronecker_factor_list, eigenbasis_list, strict=True):
        # If the tensor is empty, propagate an empty tensor to the output lists.
        if kronecker_factor.numel() == 0:
            casted_matrix_list.append(torch.empty(0, device=kronecker_factor.device))
            casted_eigenbasis_list.append(torch.empty(0, device=kronecker_factor.device))
            continue
        # Use the argument to decide whether to cast to float.
        if convert_to_float:
            casted_matrix_list.append(kronecker_factor.to(torch.float))
            casted_eigenbasis_list.append(eigenbasis.to(torch.float))
        else:
            casted_matrix_list.append(kronecker_factor)
            casted_eigenbasis_list.append(eigenbasis)

    # Cast exp_avg_sq to float in-place if needed
    if convert_to_float and exp_avg_sq.dtype != torch.float:
        exp_avg_sq = exp_avg_sq.to(torch.float)

    updated_eigenbasis_list: List[torch.Tensor] = []
    for ind, (kronecker_factor, eigenbasis) in enumerate(zip(casted_matrix_list, casted_eigenbasis_list, strict=True)):
        if kronecker_factor.numel() == 0:
            updated_eigenbasis_list.append(torch.empty(0, device=kronecker_factor.device))
            continue
        # construct an estimate of the eigenvalues using QL^T@L@QL or QR^T@R@QR, which should be close to diagonal
        # if the eigenbasis is close to the true eigenbasis of the kronecker factor (i.e. diagonalizes it)
        estimated_eigenvalue_matrix = eigenbasis.T @ kronecker_factor @ eigenbasis

        # Update eigenbasis when necessary. Update is skipped only when use_adaptive_criteria is True
        # but criteria is not met.
        if_update = True
        if use_adaptive_criteria and not _adaptive_criteria_met(
            estimated_eigenvalue_matrix=estimated_eigenvalue_matrix,
            tolerance=adaptive_update_criteria_tolerance,
        ):
            if_update = False
        if if_update:
            Q, exp_avg_sq = _orthogonal_iteration(
                estimated_eigenvalue_matrix=estimated_eigenvalue_matrix,
                kronecker_factor=kronecker_factor,
                eigenbasis=eigenbasis,
                ind=ind,
                exp_avg_sq=exp_avg_sq,
                convert_to_float=convert_to_float,
                power_iter_steps=power_iter_steps,
            )
            updated_eigenbasis_list.append(Q)
        else:
            # Do not update eigenbasis matrix
            updated_eigenbasis_list.append(eigenbasis)

    return updated_eigenbasis_list, exp_avg_sq


def _orthogonal_iteration(
    estimated_eigenvalue_matrix: torch.Tensor,
    kronecker_factor: torch.Tensor,
    eigenbasis: torch.Tensor,
    ind: int,
    exp_avg_sq: torch.Tensor,
    convert_to_float: bool = True,
    power_iter_steps: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the eigenbases of the preconditioner using power iteration and QR decomposition.

    This function performs multiple rounds of power iteration followed by QR decomposition
    to recompute the eigenbases of the preconditioner kronecker factor.

    Args:
        estimated_eigenvalue_matrix : Projection of kronecker factor onto the eigenbasis, should be close to diagonal
        kronecker_factor : Kronecker factor matrix.
        eigenbasis : Kronecker factor eigenbasis matrix.
        ind : Index for selecting dimension in the exp_avg_sq matrix to apply the sorting order over.
        exp_avg_sq : inner Adam second moment (exp_avg_sq).
        convert_to_float : If True, preconditioner matrices and their corresponding
            orthonormal matrices will be cast to float. Otherwise, they are left in
            their original type. Defaults to False.
        power_iter_steps: Number of power iteration steps to perform before QR decomposition.
            More steps can lead to better convergence but increased computation time.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Q: The updated eigenbasis
            - exp_avg_sq: The updated (sorted) inner Adam second moment
    """
    # extract eigenvalues from the diagonal of the projection of kronecker factor onto eigenbases estimate
    est_eigvals = torch.diag(estimated_eigenvalue_matrix)
    # Sort the estimated eigenvalues according to their magnitudes
    sort_idx = torch.argsort(est_eigvals, descending=True)
    # re-order the inner adam second moment
    exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)

    # Initialize power iteration after sorting the columns of the eigenbasis matrix according to the descending eigenvalues
    Q = eigenbasis[:, sort_idx]

    #  By default, perform QR decomposition with power iteration with FP32 precision
    # Perform multiple steps of power iteration
    for _ in range(power_iter_steps):
        # Project current eigenbases on kronecker factor
        Q = kronecker_factor @ Q
        # Perform QR to maintain orthogonality between iterations
        Q = torch.linalg.qr(Q).Q

    # When not converting to float, ensure that Q is in the original dtype
    if not convert_to_float:
        Q = Q.to(kronecker_factor.dtype)

    return Q, exp_avg_sq


@torch.compile  # type: ignore[misc]
def _adaptive_criteria_met(
    estimated_eigenvalue_matrix: torch.Tensor,
    tolerance: Optional[float] = None,
) -> bool:
    """Determines whether the eigenbasis for a factor matrix should be updated.

    Determines whether the eigenbasis for a factor matrix should be updated based on computing
    the approximate eigenvalues Q^T GG Q, where Q is the eigenbasis transformation matrix and
    GG is the Kronecker factor. The approximate eigenvalues update criterion is then defined as
    ||(Q^T GG Q) - diag(Q^T GG Q)||_F < tolerance * (Q^T GG Q)_F.

    Args:
        estimated_eigenvalue_matrix: Projection of kronecker factor onto the eigenbasis, should be close to diagonal
        tolerance: Tolerance threshold for the normalized off diagonal component of estimated eigenvalue matrix.

    Returns:
        perform_update: Whether to update eigenbasis this iteration
    """
    if tolerance is None:
        return True

    # check if normalized off-diagonal component is not greater than tolerance
    return not utils.eig.adaptive_early_exit_criterion(estimated_eigenvalue_matrix, tolerance)
