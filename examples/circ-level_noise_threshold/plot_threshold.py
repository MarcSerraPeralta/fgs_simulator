import pathlib

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


###########################

DISTANCES = [5, 7, 9, 11, 13, 15]
COHERENCES = [0, 1]
PROBS = {
    0: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
    1: [0.006, 0.007, 0.008, 0.01, 0.011],
}
NUM_SHOTS = 50_000
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
    for coherence in COHERENCES:
        probs = PROBS[coherence]
        y_values = []
        y_err = []

        for prob in probs:
            ds = xr.load_dataset(
                OUTPUT_NAME
                / f"d{distance}_p{prob:0.3f}_c{coherence}_probs{USE_PROBS}_n{NUM_SHOTS}.nc"
            )
            data = ds.log_flip_prob.values
            y_values.append(np.average(data))
            y_err.append(np.std(data) / np.sqrt(NUM_SHOTS))

        ax.errorbar(
            probs,
            y_values,
            yerr=y_err,
            marker="o" if coherence == 1 else "+",
            color=colors[distance],
            linestyle="--" if coherence == 1 else "-",
        )

ax.set_yscale("log")
ax.set_xscale("log")

ax.set_xlabel("physical error probability, $p$")
ax.set_ylabel("logical error probability, $p_L$")

ax.set_title("Circuit-level noise")

# create legend
for distance in DISTANCES:
    ax.errorbar([], [], color=colors[distance], label=f"d={distance}")
for coherence in COHERENCES:
    ax.errorbar(
        [],
        [],
        marker="o" if coherence == 1 else "+",
        color="black",
        label=f"c={coherence}",
        linestyle="--" if coherence == 1 else "-",
    )
ax.legend(loc="best")

fig.tight_layout()
for fmt in ["pdf", "svg", "png"]:
    fig.savefig(IMG_NAME / f"circ-level_threshold.{fmt}", format=fmt)

plt.show()
