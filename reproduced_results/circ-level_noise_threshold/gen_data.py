import time
import pathlib

import numpy as np

from fgs_sim import Setup
from fgs_sim.models import CircNoiseModel
from fgs_sim.utils import dataset_generator

###########################

DISTANCES = [3, 5, 7, 9, 11, 13, 15]
COHERENCES = [0, 1]
PROBS = {
    0: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
    1: [0.006, 0.007, 0.008, 0.010, 0.011],
}
NUM_SHOTS = 50_000
USE_PROBS = True

EXP_NAME = pathlib.Path("data")

###########################


def get_qubit_line(distance):
    qubit_line = []
    for i in range(1, distance):
        qubit_line.append(f"D{i}")
        qubit_line.append(f"A{i}")
    qubit_line.append(f"D{distance}")
    return qubit_line


def get_setup(prob, c):
    phi = np.arcsin(np.sqrt(prob))
    setup = {
        "setup": [
            {"idle": {"phi": phi, "c": c}},
            {"parity": {"phi": phi, "c": c}},
        ]
    }
    return setup


def memory_experiment_circ_level(model, num_rounds, data_init, return_prob=True):
    state = model.init_qubits(data_init)
    syndromes = []

    for _ in range(num_rounds):
        state = model.idle(state)

        state, syndrome = model.qec_round(state)
        syndromes.append(syndrome)

        state = model.idle(state)

    state, syndrome = model.final_qec_round(state)
    syndromes.append(syndrome)

    syndromes = np.array(syndromes)
    log_error_posterior = model.log_meas(state)

    return syndromes, log_error_posterior


EXP_NAME.mkdir(exist_ok=True, parents=True)

print(f"Start simulation: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
prev_time = time.time()

for coherence in COHERENCES:
    probs = PROBS[coherence]
    for prob in probs:
        setup_dict = get_setup(prob, coherence)
        setup = Setup(setup_dict)

        for distance in DISTANCES:
            qubit_line = get_qubit_line(distance)
            data_init = {q: 0 for q in qubit_line if "D" in q}
            num_rounds = distance - 1
            model = CircNoiseModel(setup, qubit_line=qubit_line)
            output_name = (
                EXP_NAME
                / f"d{distance}_p{prob:0.3f}_c{coherence}_probs{USE_PROBS}_n{NUM_SHOTS}.nc"
            )

            if output_name.exists():
                continue

            ds = dataset_generator(
                memory_experiment_circ_level,
                model=model,
                num_rounds=num_rounds,
                num_samples=NUM_SHOTS,
                data_init=data_init,
            )

            ds.assign_coords(c=coherence, p=prob)
            ds.to_netcdf(output_name)

            print(
                f"\033[F\033[Kd={distance} p={prob:0.3f} c={coherence} "
                f"runtime={int(time.time() - prev_time):d}s",
                flush=True,
            )
            prev_time = time.time()

print(
    f"\033[F\033[KEnd simulation: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
)
