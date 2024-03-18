from typing import Dict

import numpy as np
import xarray as xr

from .models import Model


def dataset_generator(
    experiment: callable,
    model: Model,
    num_rounds: int,
    data_init: Dict[str, bool],
    num_samples: int,
    verbose: bool = True,
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
    verbose
        If True, prints status bar of the dataset generation.
        Default: True.

    Returns
    -------
    ds
        Dataset with the syndromes and logical error posterior probabilities.
    """
    list_syndromes = []
    list_log_error_post = []
    idx_percent = num_samples // 100
    for i in range(num_samples):
        if verbose and (i % idx_percent == 0):
            print(f"\r{i // idx_percent}%", end="", flush=True)

        syndromes, log_error_post = experiment(
            model=model, num_rounds=num_rounds, data_init=data_init
        )

        list_syndromes.append(syndromes)
        list_log_error_post.append(log_error_post)
    print("\r100%", flush=True)

    ds = xr.Dataset(
        data_vars=dict(
            syndromes=(("shot", "qec_round", "anc_qubit"), list_syndromes),
            log_err_post=(("shot",), list_log_error_post),
            data_init=(("data_qubit",), [data_init[q] for q in model.data_qubits]),
        ),
        coords=dict(
            shot=list(range(num_samples)),
            qec_round=list(range(num_rounds + 1)),
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
