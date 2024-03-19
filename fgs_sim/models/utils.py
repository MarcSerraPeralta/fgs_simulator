from typing import Tuple
from itertools import product

import numpy as np
from scipy.special import comb
import numba as nb

from ..core import apply_op, get_prob_kraus
from .kraus import qubit_coh_noise, noisy_parity


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
    prob_0 = get_prob_kraus(state, (*kraus, c))

    # sample from phi=+phi
    sign = np.random.choice([+1, -1], p=[prob_0, 1 - prob_0])
    phi = sign * rot_angle
    c = 0.5 * (1 + sign * coherence)

    # apply operator
    matrices = qubit_coh_noise(
        qubit_id=qubit_id,
        phi=phi,
        num_qubits=num_data + 1,
    )
    out_state = apply_op(state, (*matrices, c))

    return out_state


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
    list_sign = [+1, -1]
    list_synd = [0, 1]
    combinations = list(product(list_sign, list_synd))
    probs = []
    for sign, synd in combinations:
        kraus = noisy_parity(
            qubit_ids=qubit_ids,
            synd=synd,
            phi=sign * rot_angle,
            num_qubits=num_data + 1,
        )
        c = 0.5 * 0.5 * (1 + sign * coherence)
        prob = get_prob_kraus(state, (*kraus, c))
        probs.append(prob)

    # sample syndrome from prob_0
    index = np.random.choice(range(len(combinations)), p=probs)
    sign, synd = combinations[index]

    phi = sign * rot_angle
    c = 0.5 * 0.5 * (1 + sign * coherence)

    # apply corresponding Kraus operator based on the
    # measured syndrome
    matrices = noisy_parity(
        qubit_ids=qubit_ids,
        synd=synd,
        phi=phi,
        num_qubits=num_data + 1,
    )
    out_state = apply_op(state, (*matrices, c))

    return out_state, synd


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
    # sample the circuit-level noise
    noise_q1 = np.random.choice(["IX", "XI", "XX"], p=[1 / 3, 1 / 3, 1 / 3])
    noise_q2 = np.random.choice(["IX", "XI", "XX"], p=[1 / 3, 1 / 3, 1 / 3])

    # apply noise due to circ-level noise in CNOTs
    if noise_q1 == "XX":
        state = apply_qubit_coh_noise(
            state=state,
            qubit_id=qubit_ids[0],
            rot_angle=rot_angle,
            coherence=coherence,
            num_data=num_data,
        )
    if noise_q2 == "XX":
        state = apply_qubit_coh_noise(
            state=state,
            qubit_id=qubit_ids[1],
            rot_angle=rot_angle,
            coherence=coherence,
            num_data=num_data,
        )

    # compute N for the parity check noise and the combinations
    num_maps = 2 + (noise_q1 == "IX") + (noise_q2 == "IX")
    list_k = np.arange(-num_maps, num_maps + 1, 2, dtype=int)
    list_synd = [0, 1]
    combinations = list(product(list_k, list_synd))
    probs = []

    # get probabilities for the parity kraus operators
    for k, synd in combinations:
        kraus = noisy_parity(
            qubit_ids=qubit_ids,
            synd=synd,
            phi=k * rot_angle,
            num_qubits=num_data + 1,
        )
        c = (
            0.5
            * comb(num_maps, (num_maps - k) // 2, exact=True)
            * 0.5**num_maps
            * (1 + coherence) ** ((num_maps - k) // 2)
            * (1 - coherence) ** ((num_maps + k) // 2)
        )
        prob = get_prob_kraus(state, (*kraus, c))
        probs.append(prob)

    # sample syndrome from probabilities
    index = np.random.choice(range(len(combinations)), p=probs)
    k, synd = combinations[index]

    phi = k * rot_angle
    c = (
        0.5
        * comb(num_maps, (num_maps - k) // 2, exact=True)
        * 0.5**num_maps
        * (1 + coherence) ** ((num_maps - k) // 2)
        * (1 - coherence) ** ((num_maps + k) // 2)
    )

    # apply corresponding Kraus operator based on the measured syndrome
    matrices = noisy_parity(
        qubit_ids=qubit_ids,
        synd=synd,
        phi=phi,
        num_qubits=num_data + 1,
    )
    state = apply_op(state, (*matrices, c))

    # apply noise due to circ-level noise in CNOTs
    if noise_q1 == "XI":
        state = apply_qubit_coh_noise(
            state=state,
            qubit_id=qubit_ids[0],
            rot_angle=rot_angle,
            coherence=coherence,
            num_data=num_data,
        )
    if noise_q2 == "XI":
        state = apply_qubit_coh_noise(
            state=state,
            qubit_id=qubit_ids[1],
            rot_angle=rot_angle,
            coherence=coherence,
            num_data=num_data,
        )

    return state, synd
