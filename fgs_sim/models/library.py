from typing import Tuple

import numpy as np

from .model import Model
from .utils import (
    apply_qubit_coh_noise,
    apply_ideal_parity,
    apply_noisy_parity,
    apply_noisy_parity_circ_level,
    apply_noisy_parity_exp_circ_level,
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
            # params = self.setup.param("incoming_noise", qubit)

            state = apply_qubit_coh_noise(
                state=state,
                qubit_id=self.qubit_id(qubit),
                rot_angle=self.setup.param("incoming_noise_phi", qubit),
                coherence=self.setup.param("incoming_noise_c", qubit),
                # rot_angle=params["phi"],
                # coherence=params["c"],
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
            assign_prob = params.get("assign_error_prob")
            qmeas_prob = params.get("decay_prob_half_meas")

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
            #params = self.setup.param("incoming_noise", qubit)

            state = apply_qubit_coh_noise(
                state=state,
                qubit_id=self.qubit_id(qubit),
                rot_angle=self.setup.param("incoming_noise_phi", qubit),
                coherence=self.setup.param("incoming_noise_c", qubit),
                #rot_angle=params["phi"],
                #coherence=params["c"],
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
            #params = self.setup.param("parity", parity)

            state, synd = apply_noisy_parity(
                state=state,
                qubit_ids=qubit_ids,
                #rot_angle=params["phi"],
                #coherence=params["c"],
                rot_angle=self.setup.param("parity_phi", parity),
                coherence=self.setup.param("parity_c", parity),
                num_data=self.num_data,
            )

            # add ancilla errors
            # The quantum measurement error happens after the measurement.
            assign_prob = self.setup.param("assign_error_prob", parity)
            qmeas_prob = self.setup.param("decay_prob_half_meas", parity)

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
            #params = self.setup.param("incoming_noise", qubit)

            state = apply_qubit_coh_noise(
                state=state,
                qubit_id=self.qubit_id(qubit),
                rot_angle=self.setup.param("incoming_noise_phi", qubit),
                coherence=self.setup.param("incoming_noise_c", qubit),
                #rot_angle=params["phi"],
                #coherence=params["c"],
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
            #params = self.setup.param("parity", parity)

            state, synd = apply_noisy_parity_circ_level(
                state=state,
                qubit_ids=qubit_ids,
                #rot_angle=params["phi"],
                #coherence=params["c"],
                rot_angle=self.setup.param("parity_phi", parity),
                coherence=self.setup.param("parity_c", parity),
                num_data=self.num_data,
            )

            # add ancilla errors
            # The quantum measurement error happens after the measurement.
            assign_prob = self.setup.param("assign_error_prob", parity)
            qmeas_prob = self.setup.param("decay_prob_half_meas", parity)

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
                state = apply_qubit_coh_noise(
                    state=state,
                    qubit_id=self.qubit_id(qubit),
                    rot_angle=np.arcsin(np.sqrt(param_decay)),
                    coherence=0,
                    num_data=self.num_data,
                )

        for k, parity in enumerate(self.parities):
            decay_prob = self.setup.param("decay_prob_half_meas", parity)
            if self.anc_states[k]:
                self.anc_states[k] = self.anc_states[k] ^ numba_choice(
                    [1 - decay_prob, decay_prob]
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

        # THE ORDER OF THE PARITIES NEEDS TO BE THE SAME ALSO WITH THE FINAL QEC ROUND!!!

        for k, parity in enumerate(self.parities):
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

            outcome_vector.append(noisy_outcome)

        return state, np.array(outcome_vector)


PARAM_NAMES_EXPCIRCNOISE = [
    "parity_phi_decay",
    "parity_coh_decay",
    "parity_phi_cnot",
    "parity_coh_cnot",
    "os00_0",
    "os10_0",
    "os01_0",
    "os11_0",
    "os00_1",
    "os10_1",
    "os01_1",
    "os11_1",
]


class ExpCircNoiseModel(Model):
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
            params = {n: self.setup.param(n, parity) for n in PARAM_NAMES_EXPCIRCNOISE}
            meas_params = [(k, v) for k, v in params.items() if k[:2] == "os"]

            state, synd = apply_noisy_parity_exp_circ_level(
                state=state,
                qubit_ids=qubit_ids,
                rot_angle_decay=params["parity_phi_decay"],
                coherence_decay=params["parity_coh_decay"],
                rot_angle_cnot=params["parity_phi_cnot"],
                coherence_cnot=params["parity_coh_cnot"],
                num_data=self.num_data,
            )

            # add ancilla measurement errors
            anc_state = self.anc_states[k] ^ synd
            ind = numba_choice(
                [m[1] for m in meas_params if int(m[0][-1]) == anc_state]
            )
            outcome, anc_state = map(int, meas_params[ind][0][2:4])  # e.g. "os01_0"

            self.anc_states[k] = anc_state
            outcome_vector.append(outcome)

        # incoming errors for next round
        for qubit in self.data_qubits:
            c = self.setup.param("incoming_noise_coh", qubit)
            phi = self.setup.param("incoming_noise_phi", qubit)

            state = apply_qubit_coh_noise(
                state=state,
                qubit_id=self.qubit_id(qubit),
                rot_angle=phi,
                coherence=c,
                num_data=self.num_data,
            )

        return state, np.array(outcome_vector)
