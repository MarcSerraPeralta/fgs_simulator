from typing import Tuple, Optional
from itertools import product

import random
import numpy as np
import numba as nb

# from scipy.special import comb

from ..core import apply_op, get_prob_kraus
from .kraus import qubit_coh_noise, noisy_parity, ideal_parity


@nb.njit()
def apply_qubit_coh_noise(
    state: Tuple[np.ndarray, float],
    qubit_id: int,
    rot_angle: float,
    coherence: float,
    num_data: int,
) -> Tuple[np.ndarray, float]:
    """
    Applies `qubit_coh_noise` operation to the given (unnormalized) FGS.

    The algorithm is based on Ref. [1].
    Note that the Kraus operators are applied with a given probability
    because the whole FGM is not Gaussian.

    FGS = Fermionic Gaussian State
    FGM = Fermionic Gaussian Map

    Parameters
    ----------
    state
        Unnormalized FGS before the `qubit_coh_noise` operation.
    qubit_id
        Index of the qubit to which apply the `qubit_coh_noise`.
        This value is used to create the Kraus operators.
    rot_angle
        Rotation angle of the `qubit_coh_noise` operation.
        It corresponds to $\\phi$ in Ref. [1].
    coherence
        Coherence of the noise in the `qubit_coh_noise` operation.
        It corresponds to $c$ in Ref. [1].
    num_data
        Total number of data qubits to simulate.
        Note: it does not include the extra qubit that it is added
        for the logical measurement to be a FGM.

    Returns
    -------
    out_state
        Unnormalized FGS after the `qubit_coh_noise` operation.

    References
    ----------
    [1] Suzuki, Yasunari, Keisuke Fujii, and Masato Koashi. "Efficient simulation of quantum
    error correction under coherent error based on the nonunitary free-fermionic formalism."
    Physical review letters 119.19 (2017): 190503.
    """
    # get probability of syndrome=0
    kraus = qubit_coh_noise(
        qubit_id=qubit_id,
        phi=rot_angle,
        num_qubits=num_data + 1,
    )
    c = 0.5 * (1 + coherence)
    prob = get_prob_kraus(state, (*kraus, c))

    # sample from phi=+phi
    index = numba_choice([prob, 1 - prob])
    sign = +1 if index == 0 else -1
    phi = sign * rot_angle
    c = 0.5 * (1 + sign * coherence)

    # apply operator
    if sign == -1:
        kraus = qubit_coh_noise(
            qubit_id=qubit_id,
            phi=phi,
            num_qubits=num_data + 1,
        )

    out_state = apply_op(state, (*kraus, c))

    return out_state


@nb.njit()
def apply_noisy_parity(
    state: Tuple[np.ndarray, float],
    qubit_ids: Tuple[int, int],
    rot_angle: float,
    coherence: float,
    num_data: int,
) -> Tuple[Tuple[np.ndarray, float], bool]:
    """
    Applies `noisy_parity` operation to the given (unnormalized) FGS.

    The algorithm is based on Ref. [1].
    Note that the Kraus operators are applied with a given probability
    because the whole FGM is not Gaussian.

    FGS = Fermionic Gaussian State
    FGM = Fermionic Gaussian Map

    Parameters
    ----------
    state
        Unnormalized FGS before the `noisy_parity` operation.
    qubit_ids
        Indexes of the qubits to which apply the `noisy_parity`.
        These values are used to create the Kraus operators.
    rot_angle
        Rotation angle of the `noisy_parity` operation.
        It corresponds to $\\phi$ in Ref. [1].
    coherence
        Coherence of the noise in the `noisy_parity` operation.
        It corresponds to $c$ in Ref. [1].
    num_data
        Total number of data qubits to simulate.
        Note: it does not include the extra qubit that it is added
        for the logical measurement to be a FGM.

    Returns
    -------
    out_state
        Unnormalized FGS after the `noisy_parity` operation.
    synd
        Syndrome outcome of the `noisy_parity` operation.

    References
    ----------
    [1] Suzuki, Yasunari, Keisuke Fujii, and Masato Koashi. "Efficient simulation of quantum
    error correction under coherent error based on the nonunitary free-fermionic formalism."
    Physical review letters 119.19 (2017): 190503.
    """
    # get probabilities
    # list_sign = [+1, -1]
    # list_synd = [0, 1]
    combinations = [[+1, 0], [+1, 1], [-1, 0], [-1, 1]]
    probs = []
    for sign, synd in combinations[:-1]:
        kraus = noisy_parity(
            qubit_ids=qubit_ids,
            synd=synd,
            phi=sign * rot_angle,
            num_qubits=num_data + 1,
        )

        c = 0.5 * 0.5 * (1 + sign * coherence)
        prob = get_prob_kraus(state, (*kraus, c))
        probs.append(prob)
    probs.append(1 - sum(probs))

    # sample syndrome from prob_0
    index = numba_choice(probs)
    sign, synd = combinations[index]

    phi = sign * rot_angle
    c = 0.5 * 0.5 * (1 + sign * coherence)

    # apply corresponding Kraus operator based on the
    # measured syndrome
    kraus = noisy_parity(
        qubit_ids=qubit_ids,
        synd=synd,
        phi=phi,
        num_qubits=num_data + 1,
    )

    out_state = apply_op(state, (*kraus, c))

    return out_state, synd


@nb.njit()
def apply_ideal_parity(
    state: Tuple[np.ndarray, float],
    qubit_ids: Tuple[int, int],
    num_data: int,
) -> Tuple[Tuple[np.ndarray, float], bool]:
    """
    Applies `ideal_parity` operation to the given (unnormalized) FGS.

    The algorithm is based on Ref. [1].
    Note that the Kraus operators are applied with a given probability
    because the whole FGM is not Gaussian.

    FGS = Fermionic Gaussian State
    FGM = Fermionic Gaussian Map

    Parameters
    ----------
    state
        Unnormalized FGS before the `ideal_parity` operation.
    qubit_ids
        Indexes of the qubits to which apply the `noisy_parity`.
        These values are used to create the Kraus operators.
    num_data
        Total number of data qubits to simulate.
        Note: it does not include the extra qubit that it is added
        for the logical measurement to be a FGM.

    Returns
    -------
    out_state
        Unnormalized FGS after the `ideal_parity` operation.
    synd
        Syndrome outcome of the `ideal_parity` operation.

    References
    ----------
    [1] Suzuki, Yasunari, Keisuke Fujii, and Masato Koashi. "Efficient simulation of quantum
    error correction under coherent error based on the nonunitary free-fermionic formalism."
    Physical review letters 119.19 (2017): 190503.
    """
    # get probabilities
    list_synd = [0, 1]
    kraus = ideal_parity(
        qubit_ids=qubit_ids,
        synd=0,
        num_qubits=num_data + 1,
    )

    c = 0.5
    prob = get_prob_kraus(state, (*kraus, c))

    # sample syndrome from prob_0
    index = numba_choice([prob, 1 - prob])
    synd = list_synd[index]
    c = 0.5

    # apply corresponding Kraus operator based on the
    # measured syndrome
    kraus = ideal_parity(
        qubit_ids=qubit_ids,
        synd=synd,
        num_qubits=num_data + 1,
    )

    out_state = apply_op(state, (*kraus, c))

    return out_state, synd


@nb.njit()
def apply_noisy_parity_circ_level(
    state: Tuple[np.ndarray, float],
    qubit_ids: Tuple[int, int],
    rot_angle: float,
    coherence: float,
    num_data: int,
) -> Tuple[Tuple[np.ndarray, float], bool]:
    """
    Applies `noisy_parity` operation to the given (unnormalized) FGS.

    The algorithm is based on Ref. [1].
    Note that the Kraus operators are applied with a given probability
    because the whole FGM is not Gaussian.

    FGS = Fermionic Gaussian State
    FGM = Fermionic Gaussian Map

    Parameters
    ----------
    state
        Unnormalized FGS before the `noisy_parity` operation.
    qubit_ids
        Indexes of the qubits to which apply the `noisy_parity`.
        These values are used to create the Kraus operators.
    rot_angle
        Rotation angle of the `noisy_parity` operation.
        It corresponds to $\\phi$ in Ref. [1].
    coherence
        Coherence of the noise in the `noisy_parity` operation.
        It corresponds to $c$ in Ref. [1].
    num_data
        Total number of data qubits to simulate.
        Note: it does not include the extra qubit that it is added
        for the logical measurement to be a FGM.

    Returns
    -------
    out_state
        Unnormalized FGS after the `noisy_parity` operation.
    synd
        Syndrome outcome of the `noisy_parity` operation.

    References
    ----------
    [1] Suzuki, Yasunari, Keisuke Fujii, and Masato Koashi. "Efficient simulation of quantum
    error correction under coherent error based on the nonunitary free-fermionic formalism."
    Physical review letters 119.19 (2017): 190503.
    """
    # apply noise due to circ-level noise in CNOTs
    if numba_choice([2 / 3, 1 / 3]) == 1:
        state = apply_qubit_coh_noise(
            state=state,
            qubit_id=qubit_ids[0],
            rot_angle=rot_angle,
            coherence=coherence,
            num_data=num_data,
        )
    if numba_choice([2 / 3, 1 / 3]) == 1:
        state = apply_qubit_coh_noise(
            state=state,
            qubit_id=qubit_ids[1],
            rot_angle=rot_angle,
            coherence=coherence,
            num_data=num_data,
        )

    # compute N for the parity check noise and the combinations
    num_maps = 2 + numba_choice([2 / 3, 1 / 3]) + numba_choice([2 / 3, 1 / 3])
    list_k = np.arange(-num_maps, num_maps + 1, 2, dtype=np.int32)
    list_synd = [0, 1]
    # combinations = tuple(product(list_k, list_synd))
    combinations = []
    for k in list_k:
        for synd in list_synd:
            combinations.append((k, synd))
    probs = []

    # get probabilities for the parity kraus operators
    for k, synd in combinations[:-1]:
        kraus = noisy_parity(
            qubit_ids=qubit_ids,
            synd=synd,
            phi=k * rot_angle,
            num_qubits=num_data + 1,
        )
        c = (
            0.5
            * comb(num_maps, (num_maps - k) // 2)
            * 0.5**num_maps
            * (1 + coherence) ** ((num_maps - k) // 2)
            * (1 - coherence) ** ((num_maps + k) // 2)
        )
        prob = get_prob_kraus(state, (*kraus, c))
        probs.append(prob)
    probs.append(1 - sum(probs))

    # sample syndrome from probabilities
    index = numba_choice(probs)
    k, synd = combinations[index]

    phi = k * rot_angle
    c = (
        0.5
        * comb(num_maps, (num_maps - k) // 2)
        * 0.5**num_maps
        * (1 + coherence) ** ((num_maps - k) // 2)
        * (1 - coherence) ** ((num_maps + k) // 2)
    )

    # apply corresponding Kraus operator based on the measured syndrome
    kraus = noisy_parity(
        qubit_ids=qubit_ids,
        synd=synd,
        phi=phi,
        num_qubits=num_data + 1,
    )

    state = apply_op(state, (*kraus, c))

    # apply noise due to circ-level noise in CNOTs
    if numba_choice([2 / 3, 1 / 3]) == 1:
        state = apply_qubit_coh_noise(
            state=state,
            qubit_id=qubit_ids[0],
            rot_angle=rot_angle,
            coherence=coherence,
            num_data=num_data,
        )
    if numba_choice([2 / 3, 1 / 3]) == 1:
        state = apply_qubit_coh_noise(
            state=state,
            qubit_id=qubit_ids[1],
            rot_angle=rot_angle,
            coherence=coherence,
            num_data=num_data,
        )

    return state, synd


@nb.njit()
def apply_noisy_parity_exp_circ_level(
    state: Tuple[np.ndarray, float],
    qubit_ids: Tuple[int, int],
    rot_angle_decay: float,
    coherence_decay: float,
    rot_angle_cnot: float,
    coherence_cnot: float,
    num_data: int,
) -> Tuple[Tuple[np.ndarray, float], bool]:
    """
    Applies `noisy_parity` operation to the given (unnormalized) FGS.

    The algorithm is based on Ref. [1].
    Note that the Kraus operators are applied with a given probability
    because the whole FGM is not Gaussian.

    FGS = Fermionic Gaussian State
    FGM = Fermionic Gaussian Map

    Parameters
    ----------
    state
        Unnormalized FGS before the `noisy_parity` operation.
    qubit_ids
        Indexes of the qubits to which apply the `noisy_parity`.
        These values are used to create the Kraus operators.
    rot_angle
        Rotation angle of the `noisy_parity` operation.
        It corresponds to $\\phi$ in Ref. [1].
    coherence
        Coherence of the noise in the `noisy_parity` operation.
        It corresponds to $c$ in Ref. [1].
    num_data
        Total number of data qubits to simulate.
        Note: it does not include the extra qubit that it is added
        for the logical measurement to be a FGM.

    Returns
    -------
    out_state
        Unnormalized FGS after the `noisy_parity` operation.
    synd
        Syndrome outcome of the `noisy_parity` operation.

    References
    ----------
    [1] Suzuki, Yasunari, Keisuke Fujii, and Masato Koashi. "Efficient simulation of quantum
    error correction under coherent error based on the nonunitary free-fermionic formalism."
    Physical review letters 119.19 (2017): 190503.
    """
    # apply noise due to circ-level noise in CNOTs
    if numba_choice([2 / 3, 1 / 3]) == 1:
        state = apply_qubit_coh_noise(
            state=state,
            qubit_id=qubit_ids[0],
            rot_angle=rot_angle_cnot,
            coherence=coherence_cnot,
            num_data=num_data,
        )
    if numba_choice([2 / 3, 1 / 3]) == 1:
        state = apply_qubit_coh_noise(
            state=state,
            qubit_id=qubit_ids[1],
            rot_angle=rot_angle_cnot,
            coherence=coherence_cnot,
            num_data=num_data,
        )

    # compute N for the parity check noise and the combinations
    num_maps = 2 + numba_choice([2 / 3, 1 / 3]) + numba_choice([2 / 3, 1 / 3])
    list_k_cnot = np.arange(-num_maps, num_maps + 1, 2, dtype=np.int32)
    list_synd, list_k_decay = [0, 1], [+1, -1]
    # combinations = tuple(product(list_k_cnot, list_k_decay, list_synd)) #njit error
    combinations = []
    for c in list_k_cnot:
        for d in list_k_decay:
            for s in list_synd:
                combinations.append((c, d, s))

    # get probabilities for the parity kraus operators
    probs = []
    for k_cnot, k_decay, synd in combinations[:-1]:
        kraus = noisy_parity(
            qubit_ids=qubit_ids,
            synd=synd,
            phi=k_cnot * rot_angle_cnot + k_decay * rot_angle_decay,
            num_qubits=num_data + 1,
        )
        c = (
            0.5
            * comb(num_maps, (num_maps - k_cnot) // 2)
            * 0.5**num_maps
            * (1 + coherence_cnot) ** ((num_maps - k_cnot) // 2)
            * (1 - coherence_cnot) ** ((num_maps + k_cnot) // 2)
            * (1 + coherence_decay) ** ((2 - k_decay) // 2)
            * (1 - coherence_decay) ** ((2 + k_decay) // 2)
        )
        prob = get_prob_kraus(state, (*kraus, c))
        probs.append(prob)
    probs.append(1 - sum(probs))

    # sample syndrome from probabilities
    index = numba_choice(probs)
    k_cnot, k_decay, synd = combinations[index]

    phi = k_cnot * rot_angle_cnot + k_decay * rot_angle_decay
    c = (
        0.5
        * comb(num_maps, (num_maps - k_cnot) // 2)
        * 0.5**num_maps
        * (1 + coherence_cnot) ** ((num_maps - k_cnot) // 2)
        * (1 - coherence_cnot) ** ((num_maps + k_cnot) // 2)
        * (1 + coherence_decay) ** ((2 - k_decay) // 2)
        * (1 - coherence_decay) ** ((2 + k_decay) // 2)
    )

    # apply corresponding Kraus operator based on the measured syndrome
    kraus = noisy_parity(
        qubit_ids=qubit_ids,
        synd=synd,
        phi=phi,
        num_qubits=num_data + 1,
    )

    state = apply_op(state, (*kraus, c))

    # apply noise due to circ-level noise in CNOTs
    if numba_choice([2 / 3, 1 / 3]) == 1:
        state = apply_qubit_coh_noise(
            state=state,
            qubit_id=qubit_ids[0],
            rot_angle=rot_angle_cnot,
            coherence=coherence_cnot,
            num_data=num_data,
        )
    if numba_choice([2 / 3, 1 / 3]) == 1:
        state = apply_qubit_coh_noise(
            state=state,
            qubit_id=qubit_ids[1],
            rot_angle=rot_angle_cnot,
            coherence=coherence_cnot,
            num_data=num_data,
        )

    return state, synd


@nb.njit
def numba_choice(probs):
    # Pick random weight value
    r = np.random.rand()
    total_prob = 0
    for idx, p in enumerate(probs):
        total_prob += p
        if r < total_prob:
            return idx
    return idx


@nb.njit
def comb(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))


@nb.njit
def factorial(n):
    value = 1
    for k in range(1, n + 1):
        value *= k
    return value
