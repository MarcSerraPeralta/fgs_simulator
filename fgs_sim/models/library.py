from typing import Tuple

import numpy as np

from .model import Model
from .kraus import qubit_coh_noise
from ..core import apply_op


class IncNoiseModel(Model):
    def qec_round(
        self, state: Tuple[np.ndarray, float]
    ) -> Tuple[Tuple[np.ndarray, float], np.ndarray]:
        """
        Returns the syndrome from a noisy QEC round given an FGS.
        The noise corresponds to adding rotations to the data qubits
        as specified in the `Model.setup` before the QEC round.
        The algorithm is based on Ref. [1].

        Paramters
        ---------
        state
            FGS before the (noisy) QEC round.

        Returns
        -------
        out_state
            FGS after the (noisy) QEC round.
        synd
            Syndrome outcome vector measured in the QEC round.

        References
        ----------
        [1] Suzuki, Yasunari, Keisuke Fujii, and Masato Koashi. "Efficient simulation of quantum
        error correction under coherent error based on the nonunitary free-fermionic formalism."
        Physical review letters 119.19 (2017): 190503.
        """
        for qubit in self.data_qubits:
            phi = 0.2
            c = 1
            matrices = qubit_coh_noise(
                qubit_id=self.qubit_id(qubit),
                phi=phi,
                num_qubits=self.num_data + 1,
            )
            g_norm = 0.5 * (1 + c)
            state = apply_op(state, (*matrices, g_norm))

        return super().qec_round(state)
