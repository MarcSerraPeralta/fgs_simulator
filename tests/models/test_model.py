import numpy as np

from fgs_sim.repetition_code import memory_experiment
from fgs_sim import Model


def test_model():
    model = Model(None, qubit_line=["D1", "A1", "D2", "A2", "D3"])
    data_init = {"D1": 0, "D2": 0, "D3": 0}
    num_rounds = 10

    outcomes, final_syndrome, log_error_posterior = memory_experiment(model, num_rounds, data_init)

    assert (outcomes == 0).all()
    assert (final_syndrome == 0).all()
    assert np.allclose(log_error_posterior, 0)

    return


def test_init_bistring():
    model = Model(None, qubit_line=["D1", "A1", "D2", "A2", "D3"])
    data_init = {"D1": 0, "D2": 1, "D3": 0}
    num_rounds = 3

    outcomes, final_syndrome, log_error_posterior = memory_experiment(model, num_rounds, data_init)

    # no-reset after the ancilla measurement
    assert (outcomes[0] == 1).all()
    assert (outcomes[1] == 0).all()
    assert (final_syndrome == 1).all()
    assert np.allclose(log_error_posterior, 0)

    data_init = {"D1": 0, "D2": 1, "D3": 1}
    num_rounds = 3

    outcomes, final_syndrome, log_error_posterior = memory_experiment(model, num_rounds, data_init)

    # no-reset after the ancilla measurement
    assert (outcomes[0] == np.array([True, False])).all()
    assert (outcomes[1] == np.array([False, False])).all()
    assert (final_syndrome == np.array([True, False])).all()
    assert np.allclose(log_error_posterior, 0)

    return

def test_inject_x_errors():
    model = Model(None, qubit_line=["D1", "A1", "D2", "A2", "D3"])
    data_init = {"D1": 0, "D2": 0, "D3": 0}

    model.reset_ancillas()
    state = model.init_qubits(data_init)
    model.inject_x_errors(state, ["D1", "D2", "D3"])
    state, outcome = model.qec_round(state)

    assert (outcome == 0).all()

    model.reset_ancillas()
    state = model.init_qubits(data_init)
    model.inject_x_errors(state, ["A1"])
    state, outcome = model.qec_round(state)

    assert outcome[0] == 1
    assert outcome[1] == 0

    return
