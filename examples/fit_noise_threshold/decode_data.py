import pathlib

import numpy as np
import xarray as xr

import stim
from pymatching import Matching

from fgs_sim.utils import get_defects, get_syndromes, concat_syndromes

###########################

DISTANCES = [3, 5, 7, 9]
COHERENCES = [0, 1]
PROBS = [0.005, 0.01, 0.02, 0.05]
NUM_SHOTS = 50_000
USE_PROBS = True

DATA_NAME = pathlib.Path("data")
OUTPUT_NAME = pathlib.Path("output")

###########################


def get_dem_uniform_st_edges(p, distance):
    dem_str = ""
    # space errors (first round does not have incoming errors)
    for r in range(1, distance):
        dem_str += f"error({p}) D{r*(distance-1)}\n"
        for q in range(1, distance - 1):
            dem_str += f"error({p}) D{r*(distance-1) + q} D{r*(distance-1) + q - 1}\n"
        dem_str += f"error({p}) D{r*(distance-1) + distance - 2} L0\n"
    # time errors (quantum meas errors)
    for r in range(1, distance):
        for q in range(distance - 1):
            dem_str += f"error({2*p}) D{r*(distance-1)+q} D{(r-1)*(distance-1)+q}\n"
    # time errors (assign errors)
    for r in range(2, distance):
        for q in range(distance - 1):
            dem_str += f"error({p}) D{r*(distance-1)+q} D{(r-2)*(distance-1)+q}\n"
    dem = stim.DetectorErrorModel(dem_str)
    return dem


def get_dem_uniform_stt2_edges(p, distance):
    dem_str = ""
    # space errors (first round does not have incoming errors)
    for r in range(1, distance):
        dem_str += f"error({p}) D{r*(distance-1)}\n"
        for q in range(1, distance - 1):
            dem_str += f"error({p}) D{r*(distance-1) + q} D{r*(distance-1) + q - 1}\n"
        dem_str += f"error({p}) D{r*(distance-1) + distance - 2} L0\n"
    # time errors (quantum meas errors)
    for r in range(1, distance):
        for q in range(distance - 1):
            dem_str += f"error({2*p}) D{r*(distance-1)+q} D{(r-1)*(distance-1)+q}\n"
    # time errors (assign errors)
    for r in range(2, distance):
        for q in range(distance - 1):
            dem_str += f"error({p}) D{r*(distance-1)+q} D{(r-2)*(distance-1)+q}\n"
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
            dem = get_dem_uniform_stt2_edges(prob, distance)
            mwpm = Matching(dem)

            bulk_syndromes = get_syndromes(ds.anc_outcomes)
            full_syndromes = concat_syndromes(bulk_syndromes, ds.final_syndromes)
            defects = get_defects(
                full_syndromes,
                frame=xr.DataArray(
                    np.ones(len(ds.anc_qubit)), coords=dict(anc_qubit=ds.anc_qubit)
                ),
            )
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
