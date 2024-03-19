from typing import Tuple

import numpy as np

from .model import Model
from .utils import (
    apply_qubit_coh_noise,
    apply_noisy_parity,
    apply_noisy_parity_circ_level,
)


class IncNoiseModel(Model):
    def idle(self, state: Tuple[np.ndarray, float]) -> Tuple[np.ndarray, float]:
        """
        Returns the unnormalized FGS after noisy idling given an FGS.
        For more information, read the docs of `models.utils.apply_qubit_coh_noise`.

        FGS = Fermionic Gaussian State

        Paramters
        ---------
        state
            Unnormalized FGS before the (noisy) idling.

        Returns
        -------
        out_state
            Unnormalized FGS after the (noisy) idling.
        """
        for qubit in self.data_qubits:
            params = self.setup.param("idle", qubit)

            state = apply_qubit_coh_noise(
                state=state,
                qubit_id=self.qubit_id(qubit),
                rot_angle=params["phi"],
                coherence=params["c"],
                num_data=self.num_data,
            )

        return state


class ParityNoiseModel(Model):
    def qec_round(self, state: Tuple[np.ndarray, float]) -> Tuple[np.ndarray, float]:
        """
        Returns the unnormalized FGS after noisy QEC round given an FGS.
        For more information, read the docs of `models.utils.apply_noisy_parity`.

        FGS = Fermionic Gaussian State

        Paramters
        ---------
        state
            Unnormalized FGS before the (noisy) QEC round.

        Returns
        -------
        out_state
            Unnormalized FGS after the (noisy) QEC round.
        """
        synd_vector = []
        for parity in self.parities:
            qubits = self.neighbours[parity]
            qubit_ids = [self.qubit_id(q) for q in qubits]
            params = self.setup.param("parity", parity)

            state, synd = apply_noisy_parity(
                state=state,
                qubit_ids=qubit_ids,
                rot_angle=params["phi"],
                coherence=params["c"],
                num_data=self.num_data,
            )
            synd_vector.append(synd)

        return state, np.array(synd_vector)


class PhenNoiseModel(Model):
    def idle(self, state: Tuple[np.ndarray, float]) -> Tuple[np.ndarray, float]:
        """
        Returns the unnormalized FGS after noisy idling given an FGS.
        For more information, read the docs of `models.utils.apply_qubit_coh_noise`.

        FGS = Fermionic Gaussian State

        Paramters
        ---------
        state
            Unnormalized FGS before the (noisy) idling.

        Returns
        -------
        out_state
            Unnormalized FGS after the (noisy) idling.
        """
        for qubit in self.data_qubits:
            params = self.setup.param("idle", qubit)

            state = apply_qubit_coh_noise(
                state=state,
                qubit_id=self.qubit_id(qubit),
                rot_angle=params["phi"],
                coherence=params["c"],
                num_data=self.num_data,
            )

        return state

    def qec_round(self, state: Tuple[np.ndarray, float]) -> Tuple[np.ndarray, float]:
        """
        Returns the unnormalized FGS after noisy QEC round given an FGS.
        For more information, read the docs of `models.utils.apply_noisy_parity`.

        FGS = Fermionic Gaussian State

        Paramters
        ---------
        state
            Unnormalized FGS before the (noisy) QEC round.

        Returns
        -------
        out_state
            Unnormalized FGS after the (noisy) QEC round.
        """
        synd_vector = []
        for parity in self.parities:
            qubits = self.neighbours[parity]
            qubit_ids = [self.qubit_id(q) for q in qubits]
            params = self.setup.param("parity", parity)

            state, synd = apply_noisy_parity(
                state=state,
                qubit_ids=qubit_ids,
                rot_angle=params["phi"],
                coherence=params["c"],
                num_data=self.num_data,
            )
            synd_vector.append(synd)

        return state, np.array(synd_vector)


class CircNoiseModel(Model):
    def idle(self, state: Tuple[np.ndarray, float]) -> Tuple[np.ndarray, float]:
        """
        Returns the unnormalized FGS after noisy idling given an FGS.
        For more information, read the docs of `models.utils.apply_qubit_coh_noise`.

        FGS = Fermionic Gaussian State

        Paramters
        ---------
        state
            Unnormalized FGS before the (noisy) idling.

        Returns
        -------
        out_state
            Unnormalized FGS after the (noisy) idling.
        """
        for qubit in self.data_qubits:
            params = self.setup.param("idle", qubit)

            state = apply_qubit_coh_noise(
                state=state,
                qubit_id=self.qubit_id(qubit),
                rot_angle=params["phi"],
                coherence=params["c"],
                num_data=self.num_data,
            )

        return state

    def qec_round(self, state: Tuple[np.ndarray, float]) -> Tuple[np.ndarray, float]:
        """
        Returns the unnormalized FGS after noisy QEC round given an FGS.
        For more information, read the docs of `models.utils.apply_noisy_parity`.

        FGS = Fermionic Gaussian State

        Paramters
        ---------
        state
            Unnormalized FGS before the (noisy) QEC round.

        Returns
        -------
        out_state
            Unnormalized FGS after the (noisy) QEC round.
        """
        synd_vector = []
        for parity in self.parities:
            qubits = self.neighbours[parity]
            qubit_ids = [self.qubit_id(q) for q in qubits]
            params = self.setup.param("parity", parity)

            state, synd = apply_noisy_parity_circ_level(
                state=state,
                qubit_ids=qubit_ids,
                rot_angle=params["phi"],
                coherence=params["c"],
                num_data=self.num_data,
            )
            synd_vector.append(synd)

        return state, np.array(synd_vector)
