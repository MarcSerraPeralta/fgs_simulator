from typing import Tuple

import numpy as np
import numba as nb


@nb.njit()
def apply_op(
    in_state: Tuple[np.ndarray, float],
    operator: Tuple[np.ndarray, np.ndarray, np.ndarray, float],
) -> Tuple[np.ndarray, float]:
    """
    Returns the resulting unnormalized FGS ($\\epsilon(\\rho)$) after applying the
    FGM ($\\epsilon$) to the given unnormalized FGS ($\\rho$).

    FGS = Fermionic Gaussian State
    FGM = Fermionic Gaussian Map

    The algorithm is based on Refs. [1-3].

    Parameters
    ----------
    in_state:
        Initial (unnormalized) FGS described by its covariance matrix and norm.
    operator:
        FGM described by its representation in terms of A, B, D, and C (see Ref. [2]).

    Returns
    -------
    out_state:
        (Unnormalized) FGS corresponding to applying the `operator` FGM to the `in_state` FGS.

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
    a, b, d, c = operator

    out_cov = a - b @ np.linalg.inv(in_cov - d) @ b.T
    out_norm = c * np.sqrt(np.linalg.det(in_cov - d)) * in_norm

    return out_cov, out_norm


@nb.njit()
def get_prob_kraus(
    in_state: Tuple[np.ndarray, float],
    kraus: Tuple[np.ndarray, np.ndarray, np.ndarray, float],
    eps: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """
    Returns the probability of applying the kraus operator of the FGM given
    the initial (unnormalized) FGS.

    FGS = Fermionic Gaussian State
    FGM = Fermionic Gaussian Map

    The algorithm is based on Refs. [1-3].

    Parameters
    ----------
    in_state:
        Initial (unnormalized) FGS described by its covariance matrix and norm.
    operator:
        FGM described by its representation in terms of A, B, D, and C (see Ref. [2]).
    eps:
        Maximum numerical error expected when computing the probability.

    Returns
    -------
    out_state:
        (Unnormalized) FGS corresponding to applying the `operator` FGM to the `in_state` FGS.

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
    a, b, d, c = kraus

    det = np.linalg.det(in_cov - d)
    # numerical issue where `det = =0.00000000001.. < 1`
    if det < 0 and det > -eps:
        det = 0

    prob = c * np.sqrt(det)

    # numerical issue where `prob = 1.00000000001.. > 1`
    if prob > 1 and prob < 1 + eps:
        prob = 1
    if prob < 0 and prob > -eps:
        prob = 0

    return prob


@nb.njit()
def sq_inner_product(
    state_1: Tuple[np.ndarray, float],
    state_2: Tuple[np.ndarray, float],
) -> float:
    """
    Returns the squared inner product ($\\mathrm{Tr}(\\rho_1 \\rho_2)$)
    of two (unnormalized) FGS ($\\rho_1$ and $\\rho_2$).

    FGS = Fermionic Gaussian State

    The algorithm is based on Eq. [26] of Ref. [2] and main text of Ref. [1].

    Parameters
    ----------
    state_1
        One of the (unnormalized) FGSs described by its covariance matrix and norm.
    state_2
        Other (unnormalized) FGSs described by its covariance matrix and norm.

    Returns
    -------
    norm
        Squared inner product of state1 and state2

    References
    ----------
    [1] Suzuki, Yasunari, Keisuke Fujii, and Masato Koashi. "Efficient simulation of quantum
    error correction under coherent error based on the nonunitary free-fermionic formalism."
    Physical review letters 119.19 (2017): 190503.

    [2] Bravyi, Sergey, Martin Suchara, and Alexander Vargo. "Efficient algorithms for maximum
    likelihood decoding in the surface code."
    Physical Review A 90.3 (2014): 032326.
    """
    cov_1, norm_1 = state_1
    cov_2, norm_2 = state_2
    num_qubits = cov_1.shape[0] // 2

    norm = np.sqrt(norm_1 * norm_2 * np.linalg.det(cov_1 + cov_2) / 2**num_qubits)

    return norm


def check_fgs(state: Tuple[np.ndarray, float], **kargs):
    """
    Checks if the given state is a FGS.
    It uses $M M^T <= 1$, $M = -M^T$, and $M_{ii} = 0$ (see Ref. [1]).

    FGS = Fermionic Gaussian State

    Parameters
    ----------
    state:
        A possible FGS described by its covariance matrix and norm.
    **kargs
        Extra arguments for `numpy.allclose`.

    References
    ----------
    [1] Bravyi, Sergey. "Lagrangian representation for fermionic linear optics."
    arXiv preprint quant-ph/0404180 (2004).
    """
    m, norm = state

    assert len(m.shape) == 2 and (m.shape[0] == m.shape[1])

    # avoid case where norm - numerical instabilities ~< 0
    assert norm >= 0 or np.allclose(norm, 0, **kargs)
    assert norm <= 1 or np.allclose(norm, 1, **kargs)
    assert np.allclose(m, -m.T, **kargs)
    assert np.allclose(np.diagonal(m), 0, **kargs)
    # avoid case where np.trace(m.T @ m) + numerical instabilities >~ m.shape[0]
    assert np.trace(m.T @ m) <= m.shape[0] or np.allclose(
        np.trace(m.T @ m), m.shape[0], **kargs
    )
    eigenvalues, eigenvectors = np.linalg.eig(m.T @ m)
    for eigenvalue in eigenvalues:
        assert eigenvalue >= 0 or np.allclose(eigenvalue, 0, **kargs)
        assert eigenvalue <= 1 or np.allclose(eigenvalue, 1, **kargs)

    return
