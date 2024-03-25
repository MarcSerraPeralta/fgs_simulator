import time
import pathlib

import numpy as np

from fgs_sim.repetition_code import memory_experiment
from fgs_sim import Setup
from fgs_sim.models import IncNoiseModel
from fgs_sim.utils import dataset_generator

###########################

DISTANCES = [3, 5, 7, 9]
COHERENCES = [0, 1]
PROBS = [0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.9]
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
    setup = {"setup": [{"idle": {"phi": phi, "c": c}}]}
    return setup


EXP_NAME.mkdir(exist_ok=True, parents=True)

print(f"Start simulation: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
prev_time = time.time()

for coherence in COHERENCES:
    for prob in PROBS:
        setup_dict = get_setup(prob, coherence)
        setup = Setup(setup_dict)

        for distance in DISTANCES:
            qubit_line = get_qubit_line(distance)
            data_init = {q: 0 for q in qubit_line if "D" in q}
            num_rounds = distance - 1
            model = IncNoiseModel(setup, qubit_line=qubit_line)
            output_name = (
                EXP_NAME
                / f"d{distance}_p{prob:0.3f}_c{coherence}_probs{USE_PROBS}_n{NUM_SHOTS}.nc"
            )

            if output_name.exists():
                continue

            ds = dataset_generator(
                memory_experiment,
                model=model,
                num_rounds=num_rounds,
                num_samples=NUM_SHOTS,
                data_init=data_init,
                return_probs=USE_PROBS,
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
