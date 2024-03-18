from typing import Tuple

import numpy as np

from .model import Model
from .kraus import qubit_coh_noise, noisy_parity
from ..core import apply_op, get_prob_kraus


class IncNoiseModel(Model):
    def idle(self, state: Tuple[np.ndarray, float]) -> Tuple[np.ndarray, float]:
        """
        Returns the unnormalized FGS after noisy idling given an FGS.
        The noise corresponds to adding rotations to the data qubits
        as specified in the `Model.setup` as `idle`.
        The algorithm is based on Ref. [1].
        Note that the Kraus operators are applied with a given probability
        because the whole FGM is not Gaussian.

        FGS = Fermionic Gaussian State
        FGM = Fermionic Gaussian Map

        Paramters
        ---------
        state
            Unnormalized FGS before the (noisy) idling.

        Returns
        -------
        out_state
            Unnormalized FGS after the (noisy) idling.

        References
        ----------
        [1] Suzuki, Yasunari, Keisuke Fujii, and Masato Koashi. "Efficient simulation of quantum
        error correction under coherent error based on the nonunitary free-fermionic formalism."
        Physical review letters 119.19 (2017): 190503.
        """
        for qubit in self.data_qubits:
            params = self.setup.param("idle", qubit)

            # get probability of syndrome=0
            kraus = qubit_coh_noise(
                qubit_id=self.qubit_id(qubit),
                phi=params["phi"],
                num_qubits=self.num_data + 1,
            )
            c = 0.5 * (1 + params["c"])
            prob_0 = get_prob_kraus(state, (*kraus, c))

            # sample from phi=+phi
            sign = np.random.choice([+1, -1], p=[prob_0, 1 - prob_0])
            phi = sign * params["phi"]
            c = 0.5 * (1 + sign * params["c"])

            # apply operator
            matrices = qubit_coh_noise(
                qubit_id=self.qubit_id(qubit),
                phi=phi,
                num_qubits=self.num_data + 1,
            )
            state = apply_op(state, (*matrices, c))

        return state


class ParityNoiseModel(Model):
    def qec_round(self, state: Tuple[np.ndarray, float]) -> Tuple[np.ndarray, float]:
        """
        Returns the unnormalized FGS after noisy QEC round given an FGS.
        The noise corresponds to adding rotations to the ancilla qubits
        at the beginning of the QEC cycle as specified in the `Model.setup` as `parity`.
        The algorithm is based on Ref. [1].

        FGS = Fermionic Gaussian State

        Paramters
        ---------
        state
            Unnormalized FGS before the (noisy) QEC round.

        Returns
        -------
        out_state
            Unnormalized FGS after the (noisy) QEC round.

        References
        ----------
        [1] Suzuki, Yasunari, Keisuke Fujii, and Masato Koashi. "Efficient simulation of quantum
        error correction under coherent error based on the nonunitary free-fermionic formalism."
        Physical review letters 119.19 (2017): 190503.
        """
        synd_vector = []
        for parity in self.parities:
            qubits = self.neighbours[parity]
            qubit_ids = [self.qubit_id(q) for q in qubits]
            params = self.setup.param("parity", parity)

            # get probabilities
            list_sign = [+1, +1, -1, -1]
            list_synd = [0, 1, 0, 1]
            probs = []
            for sign, synd in zip(list_sign, list_synd):
                kraus = noisy_parity(
                    qubit_ids=qubit_ids,
                    synd=synd,
                    phi=sign * params["phi"],
                    num_qubits=self.num_data + 1,
                )
                c = 0.5 * 0.5 * (1 + sign * params["c"])
                prob = get_prob_kraus(state, (*kraus, c))
                probs.append(prob)

            # sample syndrome from prob_0
            index = np.random.choice(range(4), p=probs)

            synd = list_synd[index]
            synd_vector.append(synd)

            sign = list_sign[index]
            phi = sign * params["phi"]
            c = 0.5 * 0.5 * (1 + sign * params["c"])

            # apply corresponding Kraus operator based on the
            # measured syndrome
            matrices = noisy_parity(
                qubit_ids=qubit_ids,
                synd=synd,
                phi=phi,
                num_qubits=self.num_data + 1,
            )
            state = apply_op(state, (*matrices, c))

        return state, np.array(synd_vector)


class PhenNoiseModel(Model):
    def idle(self, state: Tuple[np.ndarray, float]) -> Tuple[np.ndarray, float]:
        """
        Returns the unnormalized FGS after noisy idling given an FGS.
        The noise corresponds to adding rotations to the data qubits
        as specified in the `Model.setup` as `idle`.
        The algorithm is based on Ref. [1].
        Note that the Kraus operators are applied with a given probability
        because the whole FGM is not Gaussian.

        FGS = Fermionic Gaussian State
        FGM = Fermionic Gaussian Map

        Paramters
        ---------
        state
            Unnormalized FGS before the (noisy) idling.

        Returns
        -------
        out_state
            Unnormalized FGS after the (noisy) idling.

        References
        ----------
        [1] Suzuki, Yasunari, Keisuke Fujii, and Masato Koashi. "Efficient simulation of quantum
        error correction under coherent error based on the nonunitary free-fermionic formalism."
        Physical review letters 119.19 (2017): 190503.
        """
        for qubit in self.data_qubits:
            params = self.setup.param("idle", qubit)

            # get probability of syndrome=0
            kraus = qubit_coh_noise(
                qubit_id=self.qubit_id(qubit),
                phi=params["phi"],
                num_qubits=self.num_data + 1,
            )
            c = 0.5 * (1 + params["c"])
            prob_0 = get_prob_kraus(state, (*kraus, c))

            # sample from phi=+phi
            sign = np.random.choice([+1, -1], p=[prob_0, 1 - prob_0])
            phi = sign * params["phi"]
            c = 0.5 * (1 + sign * params["c"])

            # apply operator
            matrices = qubit_coh_noise(
                qubit_id=self.qubit_id(qubit),
                phi=phi,
                num_qubits=self.num_data + 1,
            )
            state = apply_op(state, (*matrices, c))

        return state

    def qec_round(self, state: Tuple[np.ndarray, float]) -> Tuple[np.ndarray, float]:
        """
        Returns the unnormalized FGS after noisy QEC round given an FGS.
        The noise corresponds to adding rotations to the ancilla qubits
        at the beginning of the QEC cycle as specified in the `Model.setup` as `parity`.
        The algorithm is based on Ref. [1].

        FGS = Fermionic Gaussian State

        Paramters
        ---------
        state
            Unnormalized FGS before the (noisy) QEC round.

        Returns
        -------
        out_state
            Unnormalized FGS after the (noisy) QEC round.

        References
        ----------
        [1] Suzuki, Yasunari, Keisuke Fujii, and Masato Koashi. "Efficient simulation of quantum
        error correction under coherent error based on the nonunitary free-fermionic formalism."
        Physical review letters 119.19 (2017): 190503.
        """
        synd_vector = []
        for parity in self.parities:
            qubits = self.neighbours[parity]
            qubit_ids = [self.qubit_id(q) for q in qubits]
            params = self.setup.param("parity", parity)

            # get probabilities
            list_sign = [+1, +1, -1, -1]
            list_synd = [0, 1, 0, 1]
            probs = []
            for sign, synd in zip(list_sign, list_synd):
                kraus = noisy_parity(
                    qubit_ids=qubit_ids,
                    synd=synd,
                    phi=sign * params["phi"],
                    num_qubits=self.num_data + 1,
                )
                c = 0.5 * 0.5 * (1 + sign * params["c"])
                prob = get_prob_kraus(state, (*kraus, c))
                probs.append(prob)

            # sample syndrome from prob_0
            index = np.random.choice(range(4), p=probs)

            synd = list_synd[index]
            synd_vector.append(synd)

            sign = list_sign[index]
            phi = sign * params["phi"]
            c = 0.5 * 0.5 * (1 + sign * params["c"])

            # apply corresponding Kraus operator based on the
            # measured syndrome
            matrices = noisy_parity(
                qubit_ids=qubit_ids,
                synd=synd,
                phi=phi,
                num_qubits=self.num_data + 1,
            )
            state = apply_op(state, (*matrices, c))

        return state, np.array(synd_vector)
