from absl import logging
from typing import Optional

import torch
from torch import Tensor

from llm_shower import utils

__all__ = ["eigh_with_fallback", "eig_orthogonal_iteration", "adaptive_early_exit_criterion"]


def eigh_with_fallback(
    x: Tensor,
    force_double: bool = False,
) -> tuple[Tensor, Tensor]:
    r"""torch.linalg.eigh() function with double precision fallback

    Wrapper over eigh() function. When it fails in current precision, will try to fallback to double precision. Default
    2nd argument of eigh UPLO is 'L'.

    Args:
        x: Tensor of shape (*, n, n) where "*" is zero or more batch dimensions consisting of symmetric or Hermitian matrices.
        force_double: Force double precision. Default False.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing the eigenvalues and eigenvectors of the input matrix A.
    """
    input_dtype = x.dtype
    # Check if x is already a diagonal matrix
    diag_result = _try_handle_diagonal_matrix(x)
    if diag_result is not None:
        return diag_result

    if force_double:
        logging.warning("Force double precision")
        eye_fp64 = torch.eye(
            x.shape[-1],
            device=x.device,
            dtype=torch.float64,
        )
        x = x.to(torch.float64) + eye_fp64 * 1e-30

    L, Q = torch.linalg.eigh(x)

    return L.to(input_dtype), Q.to(input_dtype)


def eig_orthogonal_iteration(
    x: Tensor,
    eigenvectors_estimate: Tensor,
    max_iterations: int = 1,
    tolerance: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """Approximately compute the eigendecomposition of a symmetric matrix by performing the orthogonal iteration algorithm.

    Given an initial estimate of the eigenvectors :math:`Q` of matrix :math:`A`, a power iteration and a QR decomposition
    is performed each iteration, i.e. :math:`Q, R = \\text{QR}(A \\cdot Q)`.
    When the initial estimate is the zero matrix, the eigendecomposition is computed using eigh_with_fallback.

    Note that if the criterion based on the estimated eigenvalues is already below or equal to the tolerance given the
    initial eigenvectors_estimate, the QR iterations will be skipped.

    Args:
        x: tensor of shape (n, n) where x is a symmetric or Hermitian matrix.
        eigenvectors_estimate: The current estimate of the eigenvectors of x. If None or a zero matrix,
            falls back to using eigh_with_fallback.
        max_iterations: The maximum number of iterations to perform. (Default: 1)
        tolerance: The tolerance for determining convergence in terms of the norm of the off-diagonal elements of the eigenvalue estimate.
            (Default: 0.01)

    Returns:
        tuple[Tensor, Tensor]: A tuple containing the estimated eigenvalues and eigenvectors matrix of the input matrix A.
    """

    # Check if x is already a diagonal matrix
    diag_result = _try_handle_diagonal_matrix(x)
    if diag_result is not None:
        return diag_result

    if eigenvectors_estimate is None or not eigenvectors_estimate.any():
        return eigh_with_fallback(x, force_double=True)

    # Perform orthogonal/simultaneous iterations (QR algorithm).
    Q = eigenvectors_estimate
    with utils.fp32_matmul_precision("highest"):
        estimated_eigenvalues = Q.T @ x @ Q
        iteration = 0
        # NOTE: This will skip the QR iterations if the criterion is already below or equal to the tolerance given the initial eigenvectors_estimate.
        while iteration < max_iterations and not adaptive_early_exit_criterion(estimated_eigenvalues, tolerance):
            power_iteration = x @ Q
            Q = torch.linalg.qr(power_iteration).Q
            estimated_eigenvalues = Q.T @ x @ Q
            iteration += 1
            # Sort eigenvalues in descending order and reorder eigenvectors accordingly
            # Sorting can help mitigate numerical instability since QR decompositions can mix the eigenvector estimates
            estimated_eigenvalues, indices = estimated_eigenvalues.diag().sort(stable=True, descending=True)
            Q = Q[:, indices]

    # check if estimated_eigenvalues is diagonal by checking the shape, if it is vector, return it as is, else extract diagonal
    if estimated_eigenvalues.shape == (x.shape[0],):
        return estimated_eigenvalues, Q
    else:
        return torch.diag(estimated_eigenvalues), Q


def adaptive_early_exit_criterion(estimated_eigenvalues: Tensor, tolerance: float) -> bool:
    """Evaluates if a criterion using estimated eigenvalues is below or equal to the tolerance.

    Let :math:`Q^T A Q =: B` be the similarity transformation of matrix :math:`A` using the matrix :math:`Q` containing
    the computed eigenvector estimates.
    The criterion based on the estimated eigenvalues is defined as :math:`\\|B - \\text{diag}(B)\\|_F \\leq \\text{tolerance} \\cdot \\|B\\|_F`.
    The tolerance hyperparameter should therefore be in the interval :math:`[0.0, 1.0]`.

    This convergence criterion can be motivated by considering :math:`A' = Q \\cdot \\text{diag}(B) \\cdot Q^T` as an approximation of :math:`A`.
    We have :math:`\\|A - A'\\|_F = \\|A - Q \\cdot \\text{diag}(B) \\cdot Q^T\\|_F = \\|Q^T A Q - \\text{diag}(B)\\|_F = \\|B - \\text{diag}(B)\\|_F`.
    Moreover, we have :math:`\\|B\\|_F = \\|Q^T A Q\\|_F = \\|A\\|_F`.
    Hence, the two relative errors are also equivalent: :math:`\\|A - A'\\|_F / \\|A\\|_F = \\|B - \\text{diag}(B)\\|_F / \\|B\\|_F`.

    Args:
        estimated_eigenvalues: The matrix B = :math:`Q^T A Q` whose eigenvalues are being estimated.
        tolerance: The tolerance for the criterion.

    Returns:
        bool: True if the criterion is below or equal to the tolerance, False otherwise.

    """
    norm = torch.linalg.norm(estimated_eigenvalues)
    diagonal_norm = torch.linalg.norm(estimated_eigenvalues.diag())
    off_diagonal_norm = torch.sqrt(norm**2 - diagonal_norm**2)
    return off_diagonal_norm <= tolerance * norm


def _is_diagonal(x: Tensor, tolerance: float = 1e-6) -> bool:
    r"""Checks if symmetric matrix is diagonal. Throw if the input is not a square matrix."""

    x_shape = x.shape
    if len(x_shape) != 2:
        raise ValueError(f"Matrix is not 2-dimensional! {x_shape=}")

    if x_shape[0] != x_shape[1]:
        raise ValueError(f"Matrix is not square! {x_shape=}")

    # Check both upper triangular part and lower triangular part are all zeros.
    return not x.triu(diagonal=1).any() and not x.tril(diagonal=-1).any()


def _try_handle_diagonal_matrix(x: Tensor) -> Optional[tuple[Tensor, Tensor]]:
    """Checks if matrix A is diagonal and returns its eigenvalues/vectors if so.

    Args:
        x: Tensor of shape (n, n) where x is a symmetric or Hermitian matrix.

    Returns:
        Optional[tuple[Tensor, Tensor]]: Sorted eigenvalues and eigenvectors if A is diagonal, None otherwise.
    """
    input_dtype = x.dtype
    if _is_diagonal(x):
        # If x is diagonal, eigenvalues are the diagonal elements and eigenvectors are the identity matrix
        eigenvalues = torch.diag(x)
        eigenvectors = torch.eye(x.shape[0], device=x.device, dtype=input_dtype)
        # Sort eigenvalues in ascending order and reorder eigenvectors accordingly
        sorted_eigenvalues, indices = eigenvalues.sort()
        sorted_eigenvectors = eigenvectors[:, indices]
        return sorted_eigenvalues, sorted_eigenvectors
    return None
