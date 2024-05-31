from typing import Tuple

import numpy as np

from .model import Model
from .utils import (
    apply_qubit_coh_noise,
    apply_noisy_parity,
    apply_noisy_parity_circ_level,
    numba_choice,
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
        outcome_vector = []
        for k, parity in enumerate(self.parities):
            qubits = self.neighbours[parity]
            qubit_ids = (self.qubit_id(qubits[0]), self.qubit_id(qubits[1]))
            params = self.setup.param("parity", parity)

            state, synd = apply_noisy_parity(
                state=state,
                qubit_ids=qubit_ids,
                rot_angle=params["phi"],
                coherence=params["c"],
                num_data=self.num_data,
            )

            # add ancilla errors
            # The quantum measurement error happens after the measurement.
            assign_prob = params.get("assing_prob")
            qmeas_prob = params.get("qmeas_prob")

            outcome = synd ^ self.anc_states[k]
            noisy_outcome = outcome ^ numba_choice([1 - assign_prob, assign_prob])
            self.anc_states[k] = (
                self.anc_states[k] ^ synd ^ numba_choice([1 - qmeas_prob, qmeas_prob])
            )

            outcome_vector.append(noisy_outcome)

        return state, np.array(outcome_vector)


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
        outcome_vector = []
        for k, parity in enumerate(self.parities):
            qubits = self.neighbours[parity]
            qubit_ids = (self.qubit_id(qubits[0]), self.qubit_id(qubits[1]))
            params = self.setup.param("parity", parity)

            state, synd = apply_noisy_parity(
                state=state,
                qubit_ids=qubit_ids,
                rot_angle=params["phi"],
                coherence=params["c"],
                num_data=self.num_data,
            )

            # add ancilla errors
            # The quantum measurement error happens after the measurement.
            assign_prob = params.get("assing_prob")
            qmeas_prob = params.get("qmeas_prob")

            outcome = synd ^ self.anc_states[k]
            noisy_outcome = outcome ^ numba_choice([1 - assign_prob, assign_prob])
            self.anc_states[k] = (
                self.anc_states[k] ^ synd ^ numba_choice([1 - qmeas_prob, qmeas_prob])
            )

            outcome_vector.append(noisy_outcome)

        return state, np.array(outcome_vector)


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
        outcome_vector = []
        for k, parity in enumerate(self.parities):
            qubits = self.neighbours[parity]
            qubit_ids = (self.qubit_id(qubits[0]), self.qubit_id(qubits[1]))
            params = self.setup.param("parity", parity)

            state, synd = apply_noisy_parity_circ_level(
                state=state,
                qubit_ids=qubit_ids,
                rot_angle=params["phi"],
                coherence=params["c"],
                num_data=self.num_data,
            )

            # add ancilla errors
            # The quantum measurement error happens after the measurement.
            assign_prob = params.get("assing_prob")
            qmeas_prob = params.get("qmeas_prob")

            outcome = synd ^ self.anc_states[k]
            noisy_outcome = outcome ^ numba_choice([1 - assign_prob, assign_prob])
            self.anc_states[k] = (
                self.anc_states[k] ^ synd ^ numba_choice([1 - qmeas_prob, qmeas_prob])
            )

            outcome_vector.append(noisy_outcome)

        return state, np.array(outcome_vector)


class FitNoiseModel(Model):
    def idle(self, state: Tuple[np.ndarray, float]) -> Tuple[np.ndarray, float]:
        """
        Returns the unnormalized FGS after noisy idling (X error + coherent X error) given an FGS.
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
            params_decay = self.setup.param("idle_decay", qubit)
            params_coh = self.setup.param("idle_coh", qubit)

            state = apply_qubit_coh_noise(
                state=state,
                qubit_id=self.qubit_id(qubit),
                rot_angle=params_coh["phi"],
                coherence=1,
                num_data=self.num_data,
            )
            state = apply_qubit_coh_noise(
                state=state,
                qubit_id=self.qubit_id(qubit),
                rot_angle=np.arcsin(np.sqrt(params_decay["prob"])),
                coherence=0,
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
        outcome_vector = []
        for k, parity in enumerate(self.parities):
            qubits = self.neighbours[parity]
            qubit_ids = (self.qubit_id(qubits[0]), self.qubit_id(qubits[1]))
            params = self.setup.param("parity", parity)

            state, synd = apply_noisy_parity_circ_level(
                state=state,
                qubit_ids=qubit_ids,
                rot_angle=params["phi"],
                coherence=params["c"],
                num_data=self.num_data,
            )

            # add ancilla errors
            # the error channel for the ancilla measurement is:
            # decay + assignment error + decay
            assign_prob = params.get("assing_prob")
            decay_prob = params.get("decay_prob")

            self.anc_states[k] = self.anc_states[k] ^ synd
            if self.anc_states[k]:
                self.anc_states[k] = self.anc_states[k] ^ numba_choice([1 - decay_prob, decay_prob])

            outcome = self.anc_states[k]
            noisy_outcome = outcome ^ numba_choice([1 - assign_prob, assign_prob])

            if self.anc_states[k]:
                self.anc_states[k] = self.anc_states[k] ^ numba_choice([1 - decay_prob, decay_prob])

            outcome_vector.append(noisy_outcome)

        return state, np.array(outcome_vector)
