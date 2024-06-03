from typing import Dict, Tuple

import random
import numpy as np

from .models import Model
from .core import check_fgs


def memory_experiment(
    model: Model,
    num_rounds: int,
    data_init: Dict[str, bool],
    return_prob: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Returns a single sample of running the memory experiment for the
    repetition code.

    Notes
    -----
    The `rot_basis` parameters is not needed as it only leads to a change
    of the Jordan-Wigner transformation used. If noise is dependent on the basis,
    it should be specified in the noise setup.

    The formalism used for this simulation takes the parity check as a full block,
    not as a sequence of gates. Therefore, the different implementations of the
    partity checks are not important and these differences can be taken into account
    in the noise model.

    Parameters
    ----------
    model
        Noise model with `idle` and `qec_round` blocks.
    num_rounds
        Number of QEC rounds to run.
    data_init
        Dictionary specifying the initial state of the data qubits.
        Note that for the FGS simulation to be efficient, the data qubits are
        initialized in a superposition of `data_init` and the oposite of `data_init`.
    return_prob
        Flag to specify the type of logical output to return.
        By default, it outputs the logical error probability.
        If False, samples a single bit corresponding to a logical error
        given the logical error probability.

    Returns
    -------
    syndrome: np.ndarray(num_rounds + 1)
        Syndrome outcomes including the final round.
    log_error_posterior
        Posterior logical error probability given the observed syndromes.
        Note that this is the average logical error probability of `data_init`
        and the oposite of `data_init`.
    """
    if not isinstance(num_rounds, int):
        raise ValueError(f"num_rounds expected as int, got {type(num_rounds)} instead.")
    if num_rounds <= 0:
        raise ValueError("num_rounds needs to be a positive integer")

    model.reset_ancillas()
    state = model.init_qubits(data_init)
    syndromes = []

    for _ in range(num_rounds):
        state = model.incoming_noise(state)
        state, syndrome = model.qec_round(state)
        syndromes.append(syndrome)
        state = model.idle(state)

    state = model.incoming_noise(state)
    state, syndrome = model.final_qec_round(state)
    syndromes.append(syndrome)

    syndromes = np.array(syndromes)
    log_error_posterior = model.log_meas(state)

    if return_prob:
        return syndromes, log_error_posterior

    log_error = random.choices(
        [0, 1], weights=[1 - log_error_posterior, log_error_posterior]
    )[0]
    return syndromes, log_error
