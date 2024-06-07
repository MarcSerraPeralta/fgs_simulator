import pathlib

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

###########################

DISTANCES = [3, 5, 7, 9, 11, 13, 15]
PROBS = [0.06, 0.07, 0.08, 0.10, 0.11, 0.12]
NUM_SHOTS = 50_000
NUM_SHOTS_STIM = 1_000_000
USE_PROBS = True

OUTPUT_NAME = pathlib.Path("output")
IMG_NAME = pathlib.Path("img")

###########################

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
            OUTPUT_NAME
            / f"d{distance}_p{prob:0.3f}_c0_probs{USE_PROBS}_n{NUM_SHOTS}.nc"
        )
        data = ds.log_flip_prob.values
        y_values.append(np.average(data))
        y_err.append(np.std(data) / np.sqrt(NUM_SHOTS))

    y_values_stim = []
    y_err_stim = []
    for prob in PROBS:
        ds = xr.load_dataset(
            OUTPUT_NAME / f"d{distance}_p{prob:0.3f}_c0_n{NUM_SHOTS_STIM}_stim.nc"
        )
        data = ds.log_flip_prob.values
        y_values_stim.append(np.average(data))
        y_err_stim.append(np.std(data) / np.sqrt(NUM_SHOTS))

    diff = np.array(y_values) - np.array(y_values_stim)
    diff_err = np.sqrt(np.array(y_err) ** 2 + np.array(y_err_stim) ** 2)

    ax.errorbar(
        PROBS,
        diff,
        yerr=diff_err,
        marker="+",
        color=colors[distance],
        linestyle="none",
    )

# ax.set_yscale("log")
# ax.set_xscale("log")

ax.set_xlabel("physical error probability, $p$")
ax.set_ylabel("difference in logical error probability, $p_L(FGS) - p_L(stim)$")

ax.set_title("Phenomenological noise")

# create legend
for distance in DISTANCES:
    ax.errorbar([], [], color=colors[distance], label=f"d={distance}")
ax.legend(loc="best")

fig.tight_layout()
for fmt in ["pdf", "svg", "png"]:
    fig.savefig(IMG_NAME / f"phenomenological_threshold_difference.{fmt}", format=fmt)

plt.show()
