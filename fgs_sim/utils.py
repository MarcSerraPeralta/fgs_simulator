from collections.abc import Callable

import numpy as np
import xarray as xr

from .models import Model


def dataset_generator(
    experiment: Callable,
    model: Model,
    num_rounds: int,
    data_init: dict[str, bool],
    num_samples: int,
    return_probs: bool = True,
    verbose: bool = True,
    **kargs,
):
    """
    Samples the experiment `num_samples` times and returns the results in
    `xarray.Dataset` format.

    Parameters
    ----------
    experiment
        Experiment to run. It must return the observed syndromes and
        the logical error posterior probability.
    model
        Noise model for the experiment.
    num_rounds
        Number of QEC rounds in the experiment
    num_samples
        Number of samples to take.
    return_probs
        Flag to specify the type of logical output to return.
        By default, it outputs the logical error probability.
        If False, samples a single bit corresponding to a logical error
        given the logical error probability.
    verbose
        If True, prints status bar of the dataset generation.
        Default: True.
    **kargs
        Extra arguments for `experiment`.

    Returns
    -------
    ds
        Dataset with the syndromes and logical error posterior probabilities.
    """
    list_anc_outcomes = []
    list_final_syndromes = []
    list_log_errors = []
    idx_percent = num_samples // 100
    for i in range(num_samples):
        if verbose and (i % idx_percent == 0):
            print(f"\r{i // idx_percent}%", end="", flush=True)

        anc_outcomes, final_syndrome, log_errors = experiment(
            model=model,
            num_rounds=num_rounds,
            data_init=data_init,
            return_prob=return_probs,
            **kargs,
        )

        list_anc_outcomes.append(anc_outcomes)
        list_final_syndromes.append(final_syndrome)
        list_log_errors.append(log_errors)
    print("\r100%", flush=True)

    ds = xr.Dataset(
        data_vars=dict(
            anc_outcomes=(("shot", "qec_round", "anc_qubit"), list_anc_outcomes),
            final_syndromes=(("shot", "anc_qubit"), list_final_syndromes),
            log_errors=(("shot",), list_log_errors),
            data_init=(("data_qubit",), [int(data_init[q]) for q in model.data_qubits]),
        ),
        coords=dict(
            shot=list(range(num_samples)),
            qec_round=list(range(1, num_rounds + 1)),
            anc_qubit=model.parities,
            data_qubit=model.data_qubits,
        ),
    )

    return ds


def get_defects(syndromes: xr.DataArray, frame: xr.DataArray = None) -> xr.DataArray:
    """
    Returns the defects from the given syndromes calculated as

    d[n] = s[n] XOR s[n-1] for n=1,...,R

    with s[0] = frame, which is 0 by default.

    Paramters
    ---------
    syndromes
        Syndrome outcomes. Must contain `qec_round` coordinate.
    frame
        Initial Pauli frame, which can be computed from the initial bistring
        and the projection matrix. Must contain a dimension corresponding
        to the ancilla qubit / parity check labels.

    Returns
    -------
    defects
        Observed defects
    """
    shifted_syn = syndromes.shift(qec_round=1, fill_value=0)

    if frame is not None:
        shifted_syn[dict(qec_round=0)] = frame

    defects = syndromes ^ shifted_syn

    return defects


def get_syndromes(
    outcomes: xr.DataArray, anc_inits: xr.DataArray = None
) -> xr.DataArray:
    """
    Returns the syndromes from the given outcomes calculated as

    s[n] = m[n] XOR m[n-1] for n=1,...,R

    with m[0] = ancilla init states, which is 0 by default.

    Paramters
    ---------
    outcomes
        Measurement outcomes. Must contain `qec_round` coordinate.
    anc_inits
        Initial ancilla bitstring. Must contain a dimension corresponding
        to the ancilla qubit / parity check labels.

    Returns
    -------
    syndromes
        Observed syndromes
    """
    shifted_out = outcomes.shift(qec_round=1, fill_value=0)

    if anc_inits is not None:
        shifted_out[dict(qec_round=0)] = anc_inits

    syndromes = outcomes ^ shifted_out

    return syndromes


def concat_syndromes(
    syndromes: xr.DataArray, final_syndromes: xr.DataArray
) -> xr.DataArray:
    """Concatenate the "bulk" syndromes and the final round syndromes."""
    final_syndromes = final_syndromes.expand_dims(
        dim=dict(qec_round=[np.max(syndromes.qec_round.values) + 1])
    )
    all_syndromes = xr.concat([syndromes, final_syndromes], dim="qec_round")
    return all_syndromes


def get_prob_log_err(predictions: np.ndarray, log_err_post: np.ndarray) -> np.ndarray:
    """
    Returns the corrected logical error posterior probability for each sample using

    p = `log_err_post`      if `prediction` == 0
      = 1 - `log_err_post`  if `prediction` == 1

    Parameters
    ----------
    predictions: np.ndarray(N)
        Predictions of the decoder about the logical flips.
    log_err_post: np.ndarray(N)
        Logical error posterior probability, i.e. probability of a logical
        error given that we have observed the syndromes.

    Returns
    -------
    prob_log_err: np.ndarray(N)
        Corrected logical error posterior probability for each shot.
    """
    # flatten just in case it has not been done after the output of MWPM.
    # If it has not been flattened, the `prob_log_err` is always approx 0.5
    predictions, log_err_post = predictions.flatten(), log_err_post.flatten()

    prob_log_err = (1 - predictions) * log_err_post + predictions * (1 - log_err_post)

    return prob_log_err
