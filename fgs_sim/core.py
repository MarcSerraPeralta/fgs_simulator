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

    The algorithm is based on Refs. [1-3]. Note that Eqs. S1 and S2 from Ref. [1] and
    Eqs. 31 and 32 from Ref. [3] are not the same as Eqs. 45 and 47 of Ref. [2].
    The equations used in this function are from Ref. [2] because they lead to
    physical probabilities, while the ones from Ref. [1] do not.

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

    tmp = np.identity(in_cov.shape[0]) + in_cov @ d

    out_cov = a + b @ np.linalg.inv(tmp) @ in_cov @ b.T
    out_norm = c * np.sqrt(np.linalg.det(tmp)) * in_norm

    return out_cov, out_norm


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

    The algorithm is based on Refs. [1-3]. Note that Eqs. S1 and S2 from Ref. [1] and
    Eqs. 31 and 32 from Ref. [3] are not the same as Eqs. 45 and 47 of Ref. [2].
    The equations used in this function are from Ref. [2] because they lead to
    physical probabilities, while the ones from Ref. [1] do not.

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

    tmp = np.identity(in_cov.shape[0]) + in_cov @ d

    prob = c * np.sqrt(np.linalg.det(tmp))

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
    It uses $M M^T <= 1$, $M = -M^T$, and $M_{ii} = 0$ (see Ref. [2]).

    FGS = Fermionic Gaussian State

    Parameters
    ----------
    state:
        A possible FGS described by its covariance matrix and norm.
    **kargs
        Extra arguments for `numpy.allclose` and `numpy.close`.

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
    m, norm = state

    assert len(m.shape) == 2 and (m.shape[0] == m.shape[1])

    assert norm >= 0
    assert np.allclose(m, -m.T, **kargs)
    assert np.allclose(np.diagonal(m), 0)
    # avoid case where np.trace(m.T @ m) + numerical instabilities >~ m.shape[0]
    assert np.trace(m.T @ m) <= m.shape[0] or np.allclose(np.trace(m.T @ m), m.shape[0])

    return
