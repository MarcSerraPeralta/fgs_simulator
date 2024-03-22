from fgs_sim.repetition_code import memory_experiment
from fgs_sim import Setup
from fgs_sim.models import PhenNoiseModel

setup = Setup({"setup": [{"idle": {"phi": 0.2, "c": 0.6},
                          "parity": {"phi": 0.2, "c": 0.6}}]})
qubit_line = ["D1", "A1", "D2", "A2", "D3", "A3", "D4", "A4", "D5"]
model = PhenNoiseModel(setup, qubit_line=qubit_line)

data_init = {q: 0 for q in qubit_line if "D" in q}
num_rounds = 50
num_samples = 1_000

for i in range(num_samples):
    if i % 100 == 0:
        print(i, end="\r")
    _, _ = memory_experiment(model, num_rounds, data_init)
