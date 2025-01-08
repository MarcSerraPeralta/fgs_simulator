from collections.abc import Sequence
from typing import Dict, Tuple, Union

import random
import numpy as np

from .models import Model


def memory_experiment(
    model: Model,
    num_rounds: int,
    data_init: Dict[str, bool],
    errors_to_inject: Sequence[Sequence[str]] | None = None,
    return_prob: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Union[int, float]]:
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
    errors_to_inject
        A list of qubit label lists that specify to which qubits to inject Pauli X
        errors before each QEC cycle. It must fulfill `len(errors_to_inject) == num_rounds + 1`.
        By default, no X errors are injected.
    return_prob
        Flag to specify the type of logical output to return.
        By default, it outputs the logical error probability.
        If False, samples a single bit corresponding to a logical error
        given the logical error probability.

    Returns
    -------
    outcomes : np.ndarray(num_rounds)
        Ancilla measurement outcome from the QEC cycles.
    final_syndrome : np.ndarray
        Syndrome outcomes for the final round.
    log_error_posterior
        Posterior logical error probability given the observed outcomes.
        Note that this is the average logical error probability of `data_init`
        and the oposite of `data_init`.
    """
    if not isinstance(num_rounds, int):
        raise TypeError(f"num_rounds expected as int, got {type(num_rounds)} instead.")
    if num_rounds <= 0:
        raise ValueError("num_rounds needs to be a positive integer")
    if errors_to_inject is None:
        errors_to_inject = [[] for _ in range(num_rounds + 1)]
    if not isinstance(errors_to_inject, Sequence):
        raise TypeError(
            f"'errors_to_inject' must be iterable, but {type(errors_to_inject)} was given."
        )
    if len(errors_to_inject) != num_rounds + 1:
        raise ValueError(
            f"'errors_to_inject' must be of length `num_rounds+1`, but it was {len(errors_to_inject)}."
        )

    model.reset_ancillas()
    state = model.init_qubits(data_init)
    outcomes = []

    for k in range(num_rounds):
        state = model.incoming_noise(state)
        state = model.inject_x_errors(state, qubits=errors_to_inject[k])
        state, outcome = model.qec_round(state)
        outcomes.append(outcome)
        state = model.idle(state)

    state = model.incoming_noise(state)
    state = model.inject_x_errors(state, qubits=errors_to_inject[num_rounds])
    state, final_syndrome = model.final_qec_round(state)

    outcomes = np.array(outcomes)
    final_syndrome = np.array(final_syndrome)
    log_error_posterior = model.log_meas(state)

    if return_prob:
        return outcomes, final_syndrome, log_error_posterior

    log_error = random.choices(
        [0, 1], weights=[1 - log_error_posterior, log_error_posterior]
    )[0]
    return outcomes, final_syndrome, log_error
