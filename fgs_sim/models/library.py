from typing import Tuple

import numpy as np

from .model import Model
from .utils import (
    apply_qubit_coh_noise,
    apply_ideal_parity,
    apply_noisy_parity,
    apply_noisy_parity_circ_level,
    numba_choice,
)


class IncNoiseModel(Model):
    def incoming_noise(
        self, state: Tuple[np.ndarray, float]
    ) -> Tuple[np.ndarray, float]:
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
    def incoming_noise(
        self, state: Tuple[np.ndarray, float]
    ) -> Tuple[np.ndarray, float]:
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
    def incoming_noise(
        self, state: Tuple[np.ndarray, float]
    ) -> Tuple[np.ndarray, float]:
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
            param_decay = self.setup.param("decay_prob_half_meas", qubit)
            param_coh = self.setup.param("coh_phi", qubit)

            if param_coh != 0:
                state = apply_qubit_coh_noise(
                    state=state,
                    qubit_id=self.qubit_id(qubit),
                    rot_angle=param_coh,
                    coherence=1,
                    num_data=self.num_data,
                )
            if param_decay != 0:
                state = apply_qubit_coh_noise(
                    state=state,
                    qubit_id=self.qubit_id(qubit),
                    rot_angle=np.arcsin(np.sqrt(param_decay)),
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

        parities_and_idx = [(k, p) for k, p in enumerate(self.parities)]
        ordered_parities = parities_and_idx[::2] + parities_and_idx[1::2]

        for k, parity in ordered_parities:
            qubits = self.neighbours[parity]
            qubit_ids = (self.qubit_id(qubits[0]), self.qubit_id(qubits[1]))
            param_phi = self.setup.param("coh_phi", parity)

            if param_phi != 0:
                state, synd = apply_noisy_parity(
                    state=state,
                    qubit_ids=qubit_ids,
                    rot_angle=param_phi,
                    coherence=1,
                    num_data=self.num_data,
                )
            else:
                state, synd = apply_ideal_parity(
                    state=state,
                    qubit_ids=qubit_ids,
                    num_data=self.num_data,
                )

            # add ancilla errors
            # the error channel for the ancilla measurement is:
            # decay + assignment error + decay
            assign_prob = self.setup.param("assign_error_prob", parity)
            decay_prob = self.setup.param("decay_prob_half_meas", parity)

            self.anc_states[k] = self.anc_states[k] ^ synd

            if self.anc_states[k]:
                self.anc_states[k] = self.anc_states[k] ^ numba_choice(
                    [1 - decay_prob, decay_prob]
                )

            outcome = self.anc_states[k]
            noisy_outcome = outcome ^ numba_choice([1 - assign_prob, assign_prob])

            if self.anc_states[k]:
                self.anc_states[k] = self.anc_states[k] ^ numba_choice(
                    [1 - decay_prob, decay_prob]
                )

            outcome_vector.append(noisy_outcome)

        return state, np.array(outcome_vector)
