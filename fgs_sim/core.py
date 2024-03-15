from typing import Tuple

import numpy as np
import numba as nb


@nb.njit()
def apply_op(
    in_state: Tuple[np.ndarray, float],
    operator: Tuple[np.ndarray, np.ndarray, np.ndarray, float],
) -> Tuple[np.ndarray, float]:
    """
    Returns the resulting (pure) FGS ($G\\ket{\\Phi}$) after applying the
    FGO (G) to the given (pure) FGS ($\\ket{\\Phi}$).

    FGS = Fermionic Gaussian State
    FGO = Fermionic Gaussian Operator

    The algorithm is based on Refs. [1-3].

    Parameters
    ----------
    in_state:
        Initial FGS described by its covariance matrix and norm.
    operator:
        FGO described by its representation in terms of A, B, D, and norm
        as derived in Ref. [1-3].

    Returns
    -------
    out_state:
        FGS corresponding to applying the `operator` FGO to the `in_state` FGS.

    References
    ----------
    [1] Suzuki, Yasunari, Keisuke Fujii, and Masato Koashi. "Efficient simulation of quantum
    error correction under coherent error based on the nonunitary free-fermionic formalism."
    Physical review letters 119.19 (2017): 190503.

    [2] Bravyi, Sergey. "Lagrangian representation for fermionic linear optics."
    arXiv preprint quant-ph/0404180 (2004).

    [3] Bravyi, Sergey, Martin Suchara, and Alexander Vargo. "Efficient algorithms for maximum
    likelihood decoding in the surface code."
    Physical Review A 90.3 (2014): 032326.
    """
    in_cov, in_norm = in_state
    a, b, d, g_norm = operator

    out_cov = a - b @ np.linalg.inv(in_cov - d) @ b.T
    out_norm = g_norm * in_norm * np.sqrt(np.linalg.det(in_cov - d))

    return out_cov, out_norm


@nb.njit()
def inner_product(
    state_1: Tuple[np.ndarray, float],
    state_2: Tuple[np.ndarray, float],
) -> float:
    """
    Returns the absolute-value inner product ($|\\bra{\\Psi}\\ket{\\Phi}|$) of
    two (pure) FGS ($\\ket{\\Phi}$ and $\\ket{\\Psi}$).

    FGS = Fermionic Gaussian State

    The algorithm is based on Refs. [1-3].

    Parameters
    ----------
    state1:
        One of the FGSs described by its covariance matrix and norm.
    state2:
        Other FGSs described by its covariance matrix and norm.

    Returns
    -------
    norm:
        Absolute-value inner product of state1 and state2

    References
    ----------
    [1] Suzuki, Yasunari, Keisuke Fujii, and Masato Koashi. "Efficient simulation of quantum
    error correction under coherent error based on the nonunitary free-fermionic formalism."
    Physical review letters 119.19 (2017): 190503.

    [2] Bravyi, Sergey. "Lagrangian representation for fermionic linear optics."
    arXiv preprint quant-ph/0404180 (2004).

    [3] Bravyi, Sergey, Martin Suchara, and Alexander Vargo. "Efficient algorithms for maximum
    likelihood decoding in the surface code."
    Physical Review A 90.3 (2014): 032326.
    """
    cov_1, norm_1 = state_1
    cov_2, norm_2 = state_2
    num_qubits = cov_1.shape[0] // 2

    norm = np.sqrt(norm_1 * norm_2 * np.linalg.det(cov_1 + cov_2) / 2**num_qubits)

    return norm


@nb.njit()
def check_pure_state(state: Tuple[np.ndarray, float], **kargs):
    """
    Checks if the given FGS is a pure FGS.
    It uese $M M^T = 1$ (see Refs. [1-3]).

    FGS = Fermionic Gaussian State

    Parameters
    ----------
    state:
        An FGS described by its covariance matrix and norm.
    **kargs
        Extra arguments for `numpy.allclose` when checking $M M^T = 1$.

    References
    ----------
    [1] Suzuki, Yasunari, Keisuke Fujii, and Masato Koashi. "Efficient simulation of quantum
    error correction under coherent error based on the nonunitary free-fermionic formalism."
    Physical review letters 119.19 (2017): 190503.

    [2] Bravyi, Sergey. "Lagrangian representation for fermionic linear optics."
    arXiv preprint quant-ph/0404180 (2004).

    [3] Bravyi, Sergey, Martin Suchara, and Alexander Vargo. "Efficient algorithms for maximum
    likelihood decoding in the surface code."
    Physical Review A 90.3 (2014): 032326.
    """
    assert len(M.shape) == 2 and (M.shape[0] == M.shape[1])

    assert np.allclose(M @ M.T, np.identity(M.shape[0]), **kargs)

    return
