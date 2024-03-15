from typing import Tuple

import numpy as np

from .model import Model
from .kraus import qubit_coh_noise
from ..core import apply_op


class IncNoiseModel(Model):
    def idle(self, state: Tuple[np.ndarray, float]) -> Tuple[np.ndarray, float]:
        """
        Returns the FGS after noisy idling given an FGS.
        The noise corresponds to adding rotations to the data qubits
        as specified in the `Model.setup` as `idle`.
        The algorithm is based on Ref. [1].

        Paramters
        ---------
        state
            FGS before the (noisy) idling.

        Returns
        -------
        out_state
            FGS after the (noisy) idling.

        References
        ----------
        [1] Suzuki, Yasunari, Keisuke Fujii, and Masato Koashi. "Efficient simulation of quantum
        error correction under coherent error based on the nonunitary free-fermionic formalism."
        Physical review letters 119.19 (2017): 190503.
        """
        for qubit in self.data_qubits:
            params = self.setup.param("idle", qubit)
            matrices = qubit_coh_noise(
                qubit_id=self.qubit_id(qubit),
                phi=params["phi"],
                num_qubits=self.num_data + 1,
            )
            g_norm = 0.5 * (1 + params["c"])
            state = apply_op(state, (*matrices, g_norm))

        return state
