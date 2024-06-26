import pathlib

import numpy as np
import xarray as xr

import stim
from pymatching import Matching

from fgs_sim.utils import get_defects, get_syndromes

###########################

DISTANCES = [3, 5, 7, 9, 11, 13, 15]
COHERENCES = [0, 1]
PROBS = [0.06, 0.07, 0.08, 0.10, 0.11, 0.12]
NUM_SHOTS = 50_000
USE_PROBS = True

DATA_NAME = pathlib.Path("data")
OUTPUT_NAME = pathlib.Path("output")

###########################


def get_dem_uniform_st_edges(distance):
    dem_str = ""
    # space errors
    for r in range(distance):
        dem_str += f"error(0.1) D{r*(distance-1)}\n"
        for q in range(1, distance - 1):
            dem_str += f"error(0.1) D{r*(distance-1) + q} D{r*(distance-1) + q - 1}\n"
        dem_str += f"error(0.1) D{r*(distance-1) + distance - 2} L0\n"
    # time errors
    for r in range(1, distance):
        for q in range(distance - 1):
            dem_str += f"error(0.1) D{r*(distance-1)+q} D{(r-1)*(distance-1)+q}\n"
    dem = stim.DetectorErrorModel(dem_str)
    return dem


OUTPUT_NAME.mkdir(exist_ok=True, parents=True)

for coherence in COHERENCES:
    for prob in PROBS:
        for distance in DISTANCES:
            file_name = f"d{distance}_p{prob:0.3f}_c{coherence}_probs{USE_PROBS}_n{NUM_SHOTS}.nc"
            output_name = OUTPUT_NAME / file_name
            if output_name.exists():
                continue

            ds = xr.load_dataset(DATA_NAME / file_name)
            dem = get_dem_uniform_st_edges(distance)
            mwpm = Matching(dem)
            # mwpm.draw()
            # plt.show()

            defects = get_defects(get_syndromes(ds.outcomes))
            defects = defects.transpose("shot", "qec_round", "anc_qubit").values
            defect_vec = defects.reshape(NUM_SHOTS, -1)
            log_errors = ds.log_errors.transpose("shot").values

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
                f"d={distance} p={prob:0.3f} c={coherence} "
                f"p_L={np.average(log_flips)} d_r={np.average(defects)}",
                flush=True,
            )
