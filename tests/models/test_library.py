import numpy as np

from fgs_sim.repetition_code import memory_experiment
from fgs_sim import Setup
from fgs_sim.models import IncNoiseModel


def test_ideal_IncNoiseModel():
    setup = Setup({"setup": [{"idle": {"phi": 0, "c": 0.6}}]})
    model = IncNoiseModel(setup, qubit_line=["D1", "A1", "D2", "A2", "D3"])
    data_init = {"D1": 0, "D2": 0, "D3": 0}
    num_rounds = 10

    syndrome, log_error_posterior = memory_experiment(model, num_rounds, data_init)

    assert (syndrome == 0).all()
    assert np.allclose(log_error_posterior, 0)

    return


def test_noisy_IncNoiseModel():
    setup = Setup({"setup": [{"idle": {"phi": np.pi / 4, "c": 0.6}}]})
    model = IncNoiseModel(setup, qubit_line=["D1", "A1", "D2", "A2", "D3"])
    data_init = {"D1": 0, "D2": 0, "D3": 0}
    num_rounds = 50

    syndrome, log_error_posterior = memory_experiment(model, num_rounds, data_init)

    assert (syndrome != 0).any()

    return
