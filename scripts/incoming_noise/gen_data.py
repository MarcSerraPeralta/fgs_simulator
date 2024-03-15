import time
import pathlib

import numpy as np
import xarray as xr

from fgs_sim.repetition_code import memory_experiment
from fgs_sim import Model, Setup
from fgs_sim.models import IncNoiseModel
from fgs_sim.utils import dataset_generator

###########################

SETUP_FILE = "incoming_noise.yaml"
QUBIT_LINE = ["D1", "A1", "D2", "A2", "D3", "A3", "D4", "A5", "D5"]
LIST_DATA_INITS = [
    {q: 0 for q in QUBIT_LINE if "D" in q},
]
LIST_NUM_ROUNDS = list(range(10, 50, 10))
NUM_SHOTS = 50_000

EXP_NAME = pathlib.Path("foo")
DATASET_TYPE = "test"

ROOT_SEED = np.random.randint(999999)

###########################

setup = Setup.from_yaml(SETUP_FILE)
model = IncNoiseModel(setup, qubit_line=QUBIT_LINE)

root_seed_sequence = np.random.SeedSequence(ROOT_SEED)
global_seeds = iter(
    root_seed_sequence.generate_state(len(LIST_NUM_ROUNDS), dtype="uint32")
)

distance = len(LIST_DATA_INITS[0])
data_qubits = list(LIST_DATA_INITS[0].keys())
data_qubits_str = "".join([q[1:] for q in data_qubits])

print(f"Start simulation: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
prev_time = time.time()

for num_rounds in LIST_NUM_ROUNDS:
    seed_sequence = np.random.SeedSequence(next(global_seeds))
    seeds = iter(seed_sequence.generate_state(len(LIST_DATA_INITS), dtype="uint32"))

    for k, data_init in enumerate(LIST_DATA_INITS):
        seed = next(seeds)
        np.random.seed(seed)

        ds = dataset_generator(
            memory_experiment,
            model=model,
            num_rounds=num_rounds,
            num_samples=NUM_SHOTS,
            data_init=data_init,
        )

        ds.assign_coords(seed=seed)

        bistring = "".join(map(str, [data_init[q] for q in data_qubits]))
        exp_folder = (
            EXP_NAME
            / f"rep-code_d{distance}_q{data_qubits_str}_s{bistring}_r{num_rounds}_n{NUM_SHOTS}"
        )
        exp_folder.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(exp_folder / f"syndromes_{DATASET_TYPE}.nc")

        print(
            f"\033[F\033[Kr={num_rounds} bitstring={k+1}/{len(LIST_DATA_INITS)} "
            f"runtime={int(time.time() - prev_time):d}s",
            flush=True,
        )
        prev_time = time.time()

print(
    f"\033[F\033[KEnd simulation: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
)
