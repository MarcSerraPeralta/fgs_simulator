import numpy as np

from fgs_sim import Setup
from fgs_sim.models import (
    IncNoiseModel,
    ParityNoiseModel,
    PhenNoiseModel,
    CircNoiseModel,
)
from fgs_sim.core import check_fgs


def debug_memory_experiment(model, num_rounds, data_init):
    state = model.init_qubits(data_init)
    syndromes = []
    check_fgs(state)

    for _ in range(num_rounds):
        state = model.incoming_noise(state)
        check_fgs(state)
        state, syndrome = model.qec_round(state)
        syndromes.append(syndrome)
        check_fgs(state)

    state = model.incoming_noise(state)
    check_fgs(state)

    state, syndrome = model.final_qec_round(state)
    syndromes.append(syndrome)
    check_fgs(state)

    syndromes = np.array(syndromes)
    log_error_posterior = model.log_meas(state)

    assert (log_error_posterior >= 0) and (log_error_posterior <= 1)

    return syndromes, log_error_posterior


def test_ideal_IncNoiseModel():
    setup = Setup(
        {
            "setup": [
                {
                    "incoming_noise": {
                        "phi": 0,
                        "c": 0.6,
                        "assign_error_prob": 0,
                        "decay_prob_half_meas": 0,
                    }
                }
            ]
        }
    )
    model = IncNoiseModel(setup, qubit_line=["D1", "A1", "D2", "A2", "D3"])
    data_init = {"D1": 0, "D2": 0, "D3": 0}
    num_rounds = 50

    syndrome, log_error_posterior = debug_memory_experiment(
        model, num_rounds, data_init
    )

    assert (syndrome == 0).all()
    assert np.allclose(log_error_posterior, 0)

    return


def test_noisy_IncNoiseModel():
    setup = Setup(
        {
            "setup": [
                {
                    "incoming_noise": {
                        "phi": np.pi / 4,
                        "c": 0.6,
                        "assign_error_prob": 0.1,
                        "decay_prob_half_meas": 0.1,
                    }
                }
            ]
        }
    )
    model = IncNoiseModel(setup, qubit_line=["D1", "A1", "D2", "A2", "D3"])
    data_init = {"D1": 0, "D2": 0, "D3": 0}
    num_rounds = 5

    for _ in range(1_000):
        syndrome, log_error_posterior = debug_memory_experiment(
            model, num_rounds, data_init
        )

    return


def test_ideal_ParityNoiseModel():
    setup = Setup(
        {
            "setup": [
                {
                    "parity": {
                        "phi": 0,
                        "c": 1,
                        "assign_error_prob": 0,
                        "decay_prob_half_meas": 0,
                    }
                }
            ]
        }
    )
    model = ParityNoiseModel(setup, qubit_line=["D1", "A1", "D2", "A2", "D3"])
    data_init = {"D1": 0, "D2": 0, "D3": 0}
    num_rounds = 20

    syndrome, log_error_posterior = debug_memory_experiment(
        model, num_rounds, data_init
    )

    assert (syndrome == 0).all()
    assert np.allclose(log_error_posterior, 0)

    return


def test_noisy_ParityNoiseModel():
    setup = Setup(
        {
            "setup": [
                {
                    "parity": {
                        "phi": np.pi / 4,
                        "c": 0.6,
                        "assign_error_prob": 0.1,
                        "decay_prob_half_meas": 0.1,
                    }
                }
            ]
        }
    )
    model = ParityNoiseModel(setup, qubit_line=["D1", "A1", "D2", "A2", "D3"])
    data_init = {"D1": 0, "D2": 0, "D3": 0}
    num_rounds = 5

    for _ in range(1_000):
        syndrome, log_error_posterior = debug_memory_experiment(
            model, num_rounds, data_init
        )

    return


def test_ideal_PhenNoiseModel():
    setup = Setup(
        {
            "setup": [
                {"incoming_noise": {"phi": 0, "c": 0.2}},
                {
                    "parity": {
                        "phi": 0,
                        "c": 1,
                        "assign_error_prob": 0,
                        "decay_prob_half_meas": 0,
                    }
                },
            ]
        }
    )
    model = PhenNoiseModel(setup, qubit_line=["D1", "A1", "D2", "A2", "D3"])
    data_init = {"D1": 0, "D2": 0, "D3": 0}
    num_rounds = 20

    syndrome, log_error_posterior = debug_memory_experiment(
        model, num_rounds, data_init
    )

    assert (syndrome == 0).all()
    assert np.allclose(log_error_posterior, 0)

    return


def test_noisy_PhenNoiseModel():
    setup = Setup(
        {
            "setup": [
                {"incoming_noise": {"phi": 0.2, "c": 0.2}},
                {
                    "parity": {
                        "phi": 0.2,
                        "c": 0.8,
                        "assign_error_prob": 0.1,
                        "decay_prob_half_meas": 0.1,
                    }
                },
            ]
        }
    )
    model = PhenNoiseModel(setup, qubit_line=["D1", "A1", "D2", "A2", "D3"])
    data_init = {"D1": 0, "D2": 0, "D3": 0}
    num_rounds = 5

    for _ in range(1_000):
        syndrome, log_error_posterior = debug_memory_experiment(
            model, num_rounds, data_init
        )

    return


def test_ideal_CircNoiseModel():
    setup = Setup(
        {
            "setup": [
                {"incoming_noise": {"phi": 0, "c": 0.2}},
                {
                    "parity": {
                        "phi": 0,
                        "c": 1,
                        "assign_error_prob": 0,
                        "decay_prob_half_meas": 0,
                    }
                },
            ]
        }
    )
    model = CircNoiseModel(setup, qubit_line=["D1", "A1", "D2", "A2", "D3"])
    data_init = {"D1": 0, "D2": 0, "D3": 0}
    num_rounds = 20

    syndrome, log_error_posterior = debug_memory_experiment(
        model, num_rounds, data_init
    )

    assert (syndrome == 0).all()
    assert np.allclose(log_error_posterior, 0)

    return


def test_noisy_CircNoiseModel():
    setup = Setup(
        {
            "setup": [
                {"incoming_noise": {"phi": 0.2, "c": 0.2}},
                {
                    "parity": {
                        "phi": 0.2,
                        "c": 0.8,
                        "assign_error_prob": 0.1,
                        "decay_prob_half_meas": 0.1,
                    }
                },
            ]
        }
    )
    model = CircNoiseModel(setup, qubit_line=["D1", "A1", "D2", "A2", "D3"])
    data_init = {"D1": 0, "D2": 0, "D3": 0}
    num_rounds = 5

    for _ in range(1_000):
        syndrome, log_error_posterior = debug_memory_experiment(
            model, num_rounds, data_init
        )

    return
