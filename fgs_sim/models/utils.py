from typing import Tuple, Optional
from itertools import product

import random
import numpy as np
from scipy.special import comb
import numba as nb

from ..core import apply_op, get_prob_kraus
from .kraus import qubit_coh_noise, noisy_parity, ideal_parity


def apply_qubit_coh_noise(
    state: Tuple[np.ndarray, float],
    qubit_id: int,
    rot_angle: float,
    coherence: float,
    num_data: int,
    pre_kraus: Optional[dict] = {},
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
    label = ("qubit_coh_noise", qubit_id, rot_angle)
    if label in pre_kraus:
        kraus = pre_kraus[label]
    else:
        kraus = qubit_coh_noise(
            qubit_id=qubit_id,
            phi=rot_angle,
            num_qubits=num_data + 1,
        )
        pre_kraus[label] = kraus
    c = 0.5 * (1 + coherence)
    prob = get_prob_kraus(state, (*kraus, c))

    # sample from phi=+phi
    sign = random.choices([+1, -1], weights=[prob, 1 - prob])[0]
    phi = sign * rot_angle
    c = 0.5 * (1 + sign * coherence)

    # apply operator
    label = ("qubit_coh_noise", qubit_id, phi)
    if label in pre_kraus:
        kraus = pre_kraus[label]
    else:
        kraus = qubit_coh_noise(
            qubit_id=qubit_id,
            phi=phi,
            num_qubits=num_data + 1,
        )
        pre_kraus[label] = kraus

    out_state = apply_op(state, (*kraus, c))

    return out_state, pre_kraus


def apply_noisy_parity(
    state: Tuple[np.ndarray, float],
    qubit_ids: Tuple[int, int],
    rot_angle: float,
    coherence: float,
    num_data: int,
    pre_kraus: Optional[dict] = {},
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
    qubit_ids = tuple(qubit_ids)

    # get probabilities
    list_sign = [+1, -1]
    list_synd = [0, 1]
    combinations = list(product(list_sign, list_synd))
    probs = []
    for sign, synd in combinations:
        label = ("noisy_parity", qubit_ids, synd, sign * rot_angle)
        if label in pre_kraus:
            kraus = pre_kraus[label]
        else:
            kraus = noisy_parity(
                qubit_ids=qubit_ids,
                synd=synd,
                phi=sign * rot_angle,
                num_qubits=num_data + 1,
            )
            pre_kraus[label] = kraus

        c = 0.5 * 0.5 * (1 + sign * coherence)
        prob = get_prob_kraus(state, (*kraus, c))
        probs.append(prob)

    # sample syndrome from prob_0
    index = random.choices(range(len(combinations)), weights=probs)[0]
    sign, synd = combinations[index]

    phi = sign * rot_angle
    c = 0.5 * 0.5 * (1 + sign * coherence)

    # apply corresponding Kraus operator based on the
    # measured syndrome
    label = ("noisy_parity", qubit_ids, synd, phi)
    if label in pre_kraus:
        kraus = pre_kraus[label]
    else:
        kraus = noisy_parity(
            qubit_ids=qubit_ids,
            synd=synd,
            phi=phi,
            num_qubits=num_data + 1,
        )
        pre_kraus[label] = kraus

    out_state = apply_op(state, (*kraus, c))

    return out_state, synd, pre_kraus


def apply_ideal_parity(
    state: Tuple[np.ndarray, float],
    qubit_ids: Tuple[int, int],
    num_data: int,
    pre_kraus: Optional[dict] = {},
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
    qubit_ids = tuple(qubit_ids)

    # get probabilities
    list_synd = [0, 1]
    probs = []
    for synd in list_synd:
        label = ("ideal_parity", qubit_ids, synd)
        if label in pre_kraus:
            kraus = pre_kraus[label]
        else:
            kraus = ideal_parity(
                qubit_ids=qubit_ids,
                synd=synd,
                num_qubits=num_data + 1,
            )
            pre_kraus[label] = kraus

        c = 0.5
        prob = get_prob_kraus(state, (*kraus, c))
        probs.append(prob)

    # sample syndrome from prob_0
    index = random.choices(range(len(list_synd)), weights=probs)[0]
    synd = list_synd[index]
    c = 0.5

    # apply corresponding Kraus operator based on the
    # measured syndrome
    label = ("ideal_parity", qubit_ids, synd)
    if label in pre_kraus:
        kraus = pre_kraus[label]
    else:
        kraus = ideal_parity(
            qubit_ids=qubit_ids,
            synd=synd,
            num_qubits=num_data + 1,
        )
        pre_kraus[label] = kraus

    out_state = apply_op(state, (*kraus, c))

    return out_state, synd, pre_kraus


def apply_noisy_parity_circ_level(
    state: Tuple[np.ndarray, float],
    qubit_ids: Tuple[int, int],
    rot_angle: float,
    coherence: float,
    num_data: int,
    pre_kraus: Optional[dict] = {},
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
    qubit_ids = tuple(qubit_ids)

    # apply noise due to circ-level noise in CNOTs
    if random.choices(["no", "XX"], weights=[2 / 3, 1 / 3])[0] == "XX":
        state, pre_kraus = apply_qubit_coh_noise(
            state=state,
            qubit_id=qubit_ids[0],
            rot_angle=rot_angle,
            coherence=coherence,
            num_data=num_data,
            pre_kraus=pre_kraus,
        )
    if random.choices(["no", "XX"], weights=[2 / 3, 1 / 3])[0] == "XX":
        state, pre_kraus = apply_qubit_coh_noise(
            state=state,
            qubit_id=qubit_ids[1],
            rot_angle=rot_angle,
            coherence=coherence,
            num_data=num_data,
            pre_kraus=pre_kraus,
        )

    # compute N for the parity check noise and the combinations
    noise_q1 = random.choices(["no", "IX"], weights=[2 / 3, 1 / 3])[0]
    noise_q2 = random.choices(["no", "IX"], weights=[2 / 3, 1 / 3])[0]
    num_maps = 2 + (noise_q1 == "IX") + (noise_q2 == "IX")
    list_k = np.arange(-num_maps, num_maps + 1, 2, dtype=int)
    list_synd = [0, 1]
    combinations = list(product(list_k, list_synd))
    probs = []

    # get probabilities for the parity kraus operators
    for k, synd in combinations:
        label = ("noisy_parity", qubit_ids, synd, k * rot_angle)
        if label in pre_kraus:
            kraus = pre_kraus[label]
        else:
            kraus = noisy_parity(
                qubit_ids=qubit_ids,
                synd=synd,
                phi=k * rot_angle,
                num_qubits=num_data + 1,
            )
            pre_kraus[label] = kraus
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
    index = random.choices(range(len(combinations)), weights=probs)[0]
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
    label = ("noisy_parity", qubit_ids, synd, phi)
    if label in pre_kraus:
        kraus = pre_kraus[label]
    else:
        kraus = noisy_parity(
            qubit_ids=qubit_ids,
            synd=synd,
            phi=phi,
            num_qubits=num_data + 1,
        )
        pre_kraus[label] = kraus

    state = apply_op(state, (*kraus, c))

    # apply noise due to circ-level noise in CNOTs
    if random.choices(["no", "XI"], weights=[2 / 3, 1 / 3])[0] == "XI":
        state, pre_kraus = apply_qubit_coh_noise(
            state=state,
            qubit_id=qubit_ids[0],
            rot_angle=rot_angle,
            coherence=coherence,
            num_data=num_data,
            pre_kraus=pre_kraus,
        )
    if random.choices(["no", "XI"], weights=[2 / 3, 1 / 3])[0] == "XI":
        state, pre_kraus = apply_qubit_coh_noise(
            state=state,
            qubit_id=qubit_ids[1],
            rot_angle=rot_angle,
            coherence=coherence,
            num_data=num_data,
            pre_kraus=pre_kraus,
        )

    return state, synd, pre_kraus
