from absl import logging
from typing import Optional

import torch
from torch import Tensor

from llm_shower import utils

__all__ = ["eigh_with_fallback", "eig_orthogonal_iteration", "adaptive_early_exit_criteria"]


def eigh_with_fallback(
    x: Tensor,
    force_double: bool = False,
    eps: Optional[float] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> tuple[Tensor, Tensor]:
    r"""torch.linalg.eigh() function with double precision fallback

    Unified wrapper over eigh() function with automatic fallback and force double precision options.
    Automatically falls back to double precision on failure and returns eigenvalues in descending order.
    Default 2nd argument of eigh UPLO is 'L'.

    Args:
        x: Tensor of shape (*, n, n) where "*" is zero or more batch dimensions consisting of symmetric or Hermitian matrices.
        force_double: Force double precision computation. Default False.
        eps: Small offset for numerical stability. If None, uses dtype-appropriate values (1e-7 for float32, 1e-15 for float64). Default None.
        output_dtype: Desired output dtype. If None, uses input dtype. Default None.

    Returns:
        tuple[Tensor, Tensor]: Eigenvalues and eigenvectors tuple (eigenvalues in descending order).
    """
    input_dtype = x.dtype
    if output_dtype is None:
        output_dtype = input_dtype

    # Set precision-appropriate epsilon if not provided
    if eps is None:
        if x.dtype == torch.float64 or force_double:
            eps = 1e-15
        else:  # float32, float16
            eps = 1e-7

    # Check if x is already a diagonal matrix
    diag_result = _try_handle_diagonal_matrix(x)
    if diag_result is not None:
        L, Q = diag_result
        # Sort in descending order for diagonal case
        L_flipped, indices = L.sort(descending=True)
        Q_flipped = Q[:, indices]
        return (L_flipped.to(output_dtype), Q_flipped.to(output_dtype))

    # Add small identity for numerical stability
    eye = torch.eye(
        x.shape[-1],
        device=x.device,
        dtype=x.dtype,
    )
    stabilized_x = torch.addmm(x, eye, eye, alpha=eps)

    if force_double:
        logging.warning("Force double precision")
        stabilized_x = stabilized_x.to(torch.float64)

    try:
        L, Q = torch.linalg.eigh(stabilized_x)
    except (torch.linalg.LinAlgError, RuntimeError) as e:
        if not force_double:
            logging.warning(f"Falling back to double precision: {e}")
            # Fallback to higher precision if the default precision fails
            stabilized_x_fp64 = stabilized_x.to(torch.float64)
            L, Q = torch.linalg.eigh(stabilized_x_fp64)
        else:
            raise e

    # Flip order to descending (`torch.linalg.eigh` returns ascending order by default)
    L_flipped = torch.flip(L, [-1])
    Q_flipped = torch.flip(Q, [-1])
    return (L_flipped.to(output_dtype), Q_flipped.to(output_dtype))


def eig_orthogonal_iteration(
    x: Tensor,
    approx_eigenvectors: Tensor,
    max_iterations: int = 1,
    tolerance: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """Approximately compute the eigendecomposition of a symmetric matrix by performing the orthogonal iteration algorithm.


    Orthogonal or subspace iteration uses iterative power iteration and QR decomposition to update the approximated eigenvectors.
    When the initial estimate is the zero matrix, the eigendecomposition is computed using `eigh_with_fallback`.

    Based on Purifying Shampoo (https://www.arxiv.org/abs/2506.03595), we use an early exit criteria to stop the QR iterations.
    This generalizes SOAP's algorithm of 1 step of power iteration for updating the eigenbasis.

    Args:
        x: tensor of shape (n, n) where x is a symmetric or Hermitian matrix.
        approx_eigenvectors: The current estimate of the eigenvectors of x. If None or a zero matrix,
            falls back to using `eigh_with_fallback`.
        max_iterations: The maximum number of iterations to perform. (Default: 1)
        tolerance: The tolerance for determining convergence in terms of the norm of the off-diagonal elements of the approximated eigenvalues.
            (Default: 0.01)

    Returns:
        tuple[Tensor, Tensor]: A tuple containing the approximated eigenvalues and eigenvectors matrix of the input matrix A.
    """

    # Check if x is already a diagonal matrix
    diag_result = _try_handle_diagonal_matrix(x)
    if diag_result is not None:
        return diag_result

    if approx_eigenvectors is None or not approx_eigenvectors.any():
        return eigh_with_fallback(x, force_double=True)

    # Perform power iteration and QR decomposition iteratively.
    with utils.fp32_matmul_precision("highest"):
        Q = approx_eigenvectors
        approx_eigenvalues_matrix = Q.T @ x @ Q
        approx_eigenvalues = torch.diag(approx_eigenvalues_matrix)
        iteration = 0
        while iteration < max_iterations and not adaptive_early_exit_criteria(approx_eigenvalues_matrix, tolerance):
            power_iteration = x @ Q
            Q = torch.linalg.qr(power_iteration).Q
            approx_eigenvalues_matrix = Q.T @ x @ Q
            iteration += 1
            # Sort eigenvalues in descending order and reorder eigenvectors accordingly
            # Sorting can help mitigate numerical instability since QR decompositions can mix the approximated eigenvectors
            approx_eigenvalues, indices = torch.diag(approx_eigenvalues_matrix).sort(stable=True, descending=True)
            Q = Q[:, indices]

    return approx_eigenvalues, Q


def adaptive_early_exit_criteria(approx_eigenvalues_matrix: Tensor, tolerance: float) -> bool:
    """Evaluates if a criteria using approximated eigenvalues is below or equal to the tolerance.

    `approx_eigenvalues_matrix` is a matrix created from the approximated eigenvectors and the symmetric matrix that is being eigendecomposed.
    We check if the ratio of the diagonal norm to the matrix norm is greater than or equal to (1 - tolerance).

    Args:
        approx_eigenvalues_matrix: The symmetric matrix whose eigenvalues is being eigendecomposed.
        tolerance: The tolerance for the early exit criteria, the min relative error between diagonal norm and matrix norm of the approximated eigenvalues and the diagonal.

    Returns:
        bool: True if the criteria is below or equal to the tolerance, False otherwise.

    """
    matrix_norm = torch.linalg.norm(approx_eigenvalues_matrix)
    approx_eigvals = torch.diag(approx_eigenvalues_matrix)
    diagonal_norm = torch.linalg.norm(approx_eigvals)
    return diagonal_norm >= (1 - tolerance) * matrix_norm


def _is_diagonal(x: Tensor) -> bool:
    r"""Checks if symmetric matrix is diagonal. Raises an error if the input is not a square matrix."""

    x_shape = x.shape
    if len(x_shape) != 2:
        raise ValueError(f"Matrix is not 2-dimensional! {x_shape=}")

    if x_shape[0] != x_shape[1]:
        raise ValueError(f"Matrix is not square! {x_shape=}")

    # Check both upper triangular part and lower triangular part are all zeros.
    return not x.triu(diagonal=1).any() and not x.tril(diagonal=-1).any()


def _try_handle_diagonal_matrix(x: Tensor) -> Optional[tuple[Tensor, Tensor]]:
    """Checks if matrix A is diagonal and returns its eigenvalues/vectors in ascending order if so.

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
