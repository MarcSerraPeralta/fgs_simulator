from typing import Dict, Tuple

import numpy as np

from .models import Model
from .core import check_fgs


def memory_experiment(
    model: Model,
    num_rounds: int,
    data_init: Dict[str, bool],
    debug: bool = True,
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
    """
    if not isinstance(num_rounds, int):
        raise ValueError(f"num_rounds expected as int, got {type(num_rounds)} instead.")
    if num_rounds <= 0:
        raise ValueError("num_rounds needs to be a positive integer")

    state = model.init_qubits(data_init)
    syndromes = []
    if debug:
        check_fgs(state)

    for _ in range(num_rounds):
        state = model.idle(state)
        if debug:
            check_fgs(state)
        state, syndrome = model.qec_round(state)
        syndromes.append(syndrome)
        if debug:
            check_fgs(state)

    state = model.idle(state)
    if debug:
        check_fgs(state)

    state, syndrome = model.final_qec_round(state)
    syndromes.append(syndrome)
    if debug:
        check_fgs(state)

    syndromes = np.array(syndromes)
    log_error_posterior = model.log_meas(state)

    if debug:
        assert (log_error_posterior >= 0) and (log_error_posterior <= 1)

    return syndromes, log_error_posterior
