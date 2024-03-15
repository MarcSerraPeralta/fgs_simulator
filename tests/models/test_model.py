import numpy as np

from fgs_sim.repetition_code import memory_experiment
from fgs_sim import Model


def test_model():
    model = Model(None, qubit_line=["D1", "A1", "D2", "A2", "D3"])
    data_init = {"D1": 0, "D2": 0, "D3": 0}
    num_rounds = 10

    syndrome, log_error_posterior = memory_experiment(model, num_rounds, data_init)

    assert (syndrome == 0).all()
    assert np.allclose(log_error_posterior, 0)

    return
