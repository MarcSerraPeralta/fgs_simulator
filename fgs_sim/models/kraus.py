from typing import Tuple

import numpy as np


def qubit_coh_noise(
    qubit_id: int,
    phi: float,
    num_qubits: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the matrices A, B, D (see Refs. [1-3] or `core.apply_op`) from a Kraus
    operators of the form:

    $e^{\\phi c_{2i-1}c_{2i}} = \\cos(\\phi) + \\sin(\\phi)c_{2i-1}c_{2i}$

    which corresponds to

    $e^{i\\phi X_i} = \\cos(\\phi) + i\\sin(\\phi) X_i$

    when using the Jordan Wigner transformation from Ref. [1] to transform the
    Majorana modes $\\{ c_i \\}_{i = 1, ..., 2n}$ with $n$ the number of qubits.

    Note that in the references the top-left element of a matrix $M$ is $M_{1,1}$,
    while in Python it is $M[0,0]$.

    Parameters
    ----------
    qubit_id
        Index $i' = i-1$ from $e^{i\\phi X_i}$.
        Its range is $i' = 0,...,n-1$ with $n$ the number of qubits, while $i = 1,...,n$.
    phi
        Angle $\\phi$ from $e^{i\\phi X_i}$.
    num_qubits
        Number of (data) qubits in the system.

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

    a = np.zeros((2 * num_qubits, 2 * num_qubits))
    d = np.zeros((2 * num_qubits, 2 * num_qubits))
    b = np.identity(2 * num_qubits)

    # Warning: one needs to be careful with the indexes because
    # (1) `qubit_id` = $i'$ starts at 0 while $i$ in the references start at 1
    # (2) in python the first element of a matrix is `M[0,0]` while in the references
    #     it is $M_{1,1}$.
    b[2 * qubit_id + 0, 2 * qubit_id + 0] += -1 + np.cos(2 * phi)
    b[2 * qubit_id + 0, 2 * qubit_id + 1] += -np.sin(2 * phi)
    b[2 * qubit_id + 1, 2 * qubit_id + 0] += np.sin(2 * phi)
    b[2 * qubit_id + 1, 2 * qubit_id + 1] += -1 + np.cos(2 * phi)

    return a, b, d


def ideal_parity(
    qubit_ids: Tuple[int, int],
    synd: int,
    num_qubits: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the matrices A, B, D (see Refs. [1-3] or `core.apply_op`) from a Kraus
    operators of the form:

    $Id - i(-1)^s c_{2i+1} c_{2i}$

    which corresponds to

    $Id + (-1)^s Z_i Z_{i+1}$

    when using the Jordan Wigner transformation from Ref. [1] to transform the
    Majorana modes $\\{ c_i \\}_{i = 1, ..., 2n}$ with $n$ the number of qubits.

    Note that in the references the top-left element of a matrix $M$ is $M_{1,1}$,
    while in Python it is $M[0,0]$.

    Parameters
    ----------
    qubit_ids
        Indexes $i' = i-1$ and $i'+1 = i$ from $Id + (-1)^s Z_i Z_{i+1}$.
        Its range is $i' = 0,...,n-1$ with $n$ the number of qubits, while $i = 1,...,n$.
    synd
        Syndrome outcome $s$ from $Id + (-1)^s Z_i Z_{i+1}$.
    num_qubits
        Number of (data) qubits in the system.

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

    return noisy_parity(qubit_ids=qubit_ids, synd=synd, phi=0, num_qubits=num_qubits)


def noisy_parity(
    qubit_ids: Tuple[int, int],
    phi: float,
    synd: int,
    num_qubits: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the matrices A, B, D (see Refs. [1-3] or `core.apply_op`) from a Kraus
    operators of the form:

    $Id - i(-1)^s e^{-2i\\phi} c_{2i+1} c_{2i}$

    which corresponds to

    $Id + (-1)^s e^{-2i\\phi} Z_i Z_{i+1}$

    when using the Jordan Wigner transformation from Ref. [1] to transform the
    Majorana modes $\\{ c_i \\}_{i = 1, ..., 2n}$ with $n$ the number of qubits.

    Note that in the references the top-left element of a matrix $M$ is $M_{1,1}$,
    while in Python it is $M[0,0]$.

    Parameters
    ----------
    qubit_ids
        Indexes $i' = i-1$ and $i'+1 = i$ from $Id + (-1)^s e^{-2i\\phi} Z_i Z_{i+1}$.
        Its range is $i' = 0,...,n-1$ with $n$ the number of qubits, while $i = 1,...,n$.
    phi
        Angle $\\phi$ from $Id + (-1)^s e^{-2i\\phi} Z_i Z_{i+1}$.
    synd
        Syndrome outcome $s$ from $Id + (-1)^s e^{-2i\\phi} Z_i Z_{i+1}$.
    num_qubits
        Number of (data) qubits in the system.

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
    assert abs(qubit_ids[1] - qubit_ids[0]) == 1
    qubit_id = min(qubit_ids)

    a = np.zeros((2 * num_qubits, 2 * num_qubits))
    b = np.identity(2 * num_qubits)

    # Warning: one needs to be careful with the indexes because
    # (1) `qubit_id` = $i'$ starts at 0 while $i$ in the references start at 1
    # (2) in python the first element of a matrix is `M[0,0]` while in the references
    #     it is $M_{1,1}$.
    a[2 * qubit_id + 1, 2 * qubit_id + 2] = -((-1) ** synd) * np.cos(2 * phi)
    a[2 * qubit_id + 2, 2 * qubit_id + 1] = +((-1) ** synd) * np.cos(2 * phi)

    b[2 * qubit_id + 1, 2 * qubit_id + 1] += -1
    b[2 * qubit_id + 1, 2 * qubit_id + 2] += (-1) ** synd * np.sin(2 * phi)
    b[2 * qubit_id + 2, 2 * qubit_id + 1] += -((-1) ** synd) * np.sin(2 * phi)
    b[2 * qubit_id + 2, 2 * qubit_id + 2] += -1

    d = -np.copy(a)

    return a, b, d
