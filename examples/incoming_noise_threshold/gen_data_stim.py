import time
import pathlib

import numpy as np
import xarray as xr
import stim
from pymatching import Matching
import matplotlib.pyplot as plt

###########################

DISTANCES = [3, 5, 7, 9]
PROBS = [0.1, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.9]
NUM_SHOTS = 1_000_000

EXP_NAME = pathlib.Path("output")

###########################


def get_qubit_line(distance):
    qubit_line = []
    for i in range(1, distance):
        qubit_line.append(f"D{i}")
        qubit_line.append(f"A{i}")
    qubit_line.append(f"D{distance}")
    return qubit_line


def get_repetition_code_inc_noise(qubit_line, num_rounds, prob):
    # data_init = 0000...
    circuit = stim.Circuit()
    data_ids = [i for i, q in enumerate(qubit_line) if "D" in q]
    anc_ids = [i for i, q in enumerate(qubit_line) if "D" not in q]

    def t(l):
        if not isinstance(l, list):
            l = [l]
        return [stim.GateTarget(i) for i in l]

    for r in range(num_rounds):
        # idle
        circuit.append(stim.CircuitInstruction("X_ERROR", t(data_ids), [prob]))

        # QEC cycle
        for anc_id in anc_ids:
            circuit.append(stim.CircuitInstruction("R", t(anc_id)))
            circuit.append(stim.CircuitInstruction("CNOT", t([anc_id - 1, anc_id])))
            circuit.append(stim.CircuitInstruction("CNOT", t([anc_id + 1, anc_id])))
            circuit.append(stim.CircuitInstruction("M", t(anc_id)))
            if r == 0:
                circuit.append(
                    stim.CircuitInstruction("DETECTOR", [stim.target_rec(-1)])
                )
            else:
                circuit.append(
                    stim.CircuitInstruction(
                        "DETECTOR",
                        [stim.target_rec(-1), stim.target_rec(-1 - len(anc_ids))],
                    )
                )

    # idle
    circuit.append(stim.CircuitInstruction("X_ERROR", t(data_ids), [prob]))

    # ideal final round
    for anc_id in anc_ids:
        circuit.append(stim.CircuitInstruction("R", t(anc_id)))
        circuit.append(stim.CircuitInstruction("CNOT", t([anc_id - 1, anc_id])))
        circuit.append(stim.CircuitInstruction("CNOT", t([anc_id + 1, anc_id])))
        circuit.append(stim.CircuitInstruction("M", t(anc_id)))
        circuit.append(
            stim.CircuitInstruction(
                "DETECTOR",
                [stim.target_rec(-1), stim.target_rec(-1 - len(anc_ids))],
            )
        )

    # logical measurement
    circuit.append(stim.CircuitInstruction("M", t(data_ids)))
    circuit.append(
        stim.CircuitInstruction("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], [0])
    )

    return circuit


def get_dem_uniform_s_edges(distance):
    dem_str = ""
    # space errors
    for r in range(distance):
        dem_str += f"error(0.1) D{r*(distance-1)}\n"
        for q in range(1, distance - 1):
            dem_str += f"error(0.1) D{r*(distance-1) + q} D{r*(distance-1) + q - 1}\n"
        dem_str += f"error(0.1) D{r*(distance-1) + distance - 2} L0\n"
    dem = stim.DetectorErrorModel(dem_str)
    return dem


EXP_NAME.mkdir(exist_ok=True, parents=True)

print(f"Start simulation: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
prev_time = time.time()

for prob in PROBS:
    for distance in DISTANCES:
        qubit_line = get_qubit_line(distance)
        data_init = {q: 0 for q in qubit_line if "D" in q}
        num_rounds = distance - 1
        output_name = EXP_NAME / f"d{distance}_p{prob:0.3f}_c0_n{NUM_SHOTS}_stim.nc"

        if output_name.exists():
            continue

        circuit = get_repetition_code_inc_noise(
            qubit_line=qubit_line, num_rounds=num_rounds, prob=prob
        )

        if output_name.exists():
            continue

        sampler = circuit.compile_detector_sampler()
        defect_vec, log_errors = sampler.sample(
            shots=NUM_SHOTS, separate_observables=True
        )
        log_errors = log_errors.flatten()

        dem = get_dem_uniform_s_edges(distance)
        mwpm = Matching(dem)

        predictions = mwpm.decode_batch(defect_vec).flatten()

        log_flips = log_errors * (1 - predictions) + (1 - log_errors) * predictions

        # only datasets can be stored in this xarray version
        out_ds = xr.Dataset(
            data_vars=dict(log_flip_prob=(("shot",), log_flips)),
            coords=dict(
                shot=list(range(NUM_SHOTS)),
            ),
        )
        out_ds.to_netcdf(output_name)

        print(
            f"d={distance} p={prob:0.3f} c=0 d_r={np.average(defect_vec):0.3f} "
            f"runtime={int(time.time() - prev_time):d}s",
            flush=True,
        )
        prev_time = time.time()

print(
    f"\033[F\033[KEnd simulation: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
)

IMG_NAME = pathlib.Path("img")
IMG_NAME.mkdir(exist_ok=True, parents=True)

fig, ax = plt.subplots()

# Create colors for each distance
cmap = plt.cm.rainbow
colors = {d: cmap(i / (len(DISTANCES) - 1)) for i, d in enumerate(DISTANCES)}

for distance in DISTANCES:
    y_values = []
    y_err = []

    for prob in PROBS:
        ds = xr.load_dataset(
            EXP_NAME / f"d{distance}_p{prob:0.3f}_c0_n{NUM_SHOTS}_stim.nc"
        )
        data = ds.log_flip_prob.values
        y_values.append(np.average(data))
        y_err.append(np.std(data) / np.sqrt(NUM_SHOTS))

    ax.errorbar(
        PROBS,
        y_values,
        yerr=y_err,
        marker="+",
        color=colors[distance],
        linestyle="-",
    )

# ax.set_yscale("log")
# ax.set_xscale("log")

ax.set_ylim(0, 1)
ax.set_xlim(0, 1)

ax.set_xlabel("physical error probability, $p$")
ax.set_ylabel("logical error probability, $p_L$")

ax.set_title("Incoming noise")

# create legend
for distance in DISTANCES:
    ax.errorbar([], [], color=colors[distance], label=f"d={distance}")

ax.legend(loc="best")

fig.tight_layout()
for fmt in ["pdf", "svg", "png"]:
    fig.savefig(IMG_NAME / f"incoming_threshold_stim.{fmt}", format=fmt)

plt.show()
