from typing import List, Dict, Tuple

import random
import numpy as np

from .utils import apply_ideal_parity
from ..core import apply_op, get_prob_kraus


class Model:
    """
    Class for simulating QEC circuits with FGS dynamics.
    It assumes ideal QEC cycles, logical preparation and measurement.
    """

    def __init__(self, setup: Dict, qubit_line: List[str]):
        """
        Parameters
        ----------
        setup
            Setup for the noise used in the model
        qubit_line
            List of data and ancilla qubits used in the repetition code
            where the surrounding elements in the list are the nearest-
            neighbours of the qubit. For example: [D1, A1, D2]
            where A1 measures the parity of D1 and D2.

        Returns
        -------
        An instance of this class
        """
        assert len(qubit_line) % 2 == 1

        self._setup = setup
        self._qubit_line = qubit_line

        self.num_data = (len(self._qubit_line) + 1) // 2
        self._qubit_ids = {
            q: i for i, q in zip(range(self.num_data), self._qubit_line[::2])
        }
        self.data_qubits = self._qubit_line[::2]
        self.parities = self._qubit_line[1::2]
        self.neighbours = {
            p: [self.data_qubits[i], self.data_qubits[i + 1]]
            for i, p in enumerate(self.parities)
        }

        self._pre_kraus = {}

        return

    @property
    def setup(self):
        return self._setup

    @property
    def qubit_line(self):
        return self._qubit_line

    def qubit_id(self, qubit: str) -> int:
        """
        Returns the index for the specified (data) qubit.
        This index is used when creating the Kraus operators for
        the different (data) qubits.

        Parameters
        ----------
        qubit
            Label of the data qubit as specified in the `qubit_line`
            when initializing this class.

        Returns
        -------
        Index of the given (data) qubit.

        """
        if qubit not in self._qubit_ids:
            raise ValueError(
                f"Specified data qubit ('{qubit}') is not in the data qubit list: "
                + ", ".join(map(str, self._qubit_ids.values()))
            )

        return self._qubit_ids[qubit]

    def init_qubits(self, bitstring: Dict[str, bool]) -> Tuple[np.ndarray, float]:
        """
        Returns the FGS corresponding to ideal initialization to

        $\\frac{ket{b, sum(b)} + \\ket{b \\oplus 1, sum(b) \\oplus 1}}{\\sqrt{2}}$

        with b the bistring vector. This function uses the Jordan-Wigner
        transformation from Ref. [1] to transform the Majorana modes
        $\\{ c_i \\}_{i = 1, ..., 2n}$ with $n$ the number of qubits.
        The expression for the covariance matrix and norm of this FGS is in Eq. S1
        from Ref. [1] and it has been adapted using Eq. 19 of Ref. [2].

        FGS = Fermionic Gaussian State

        Paramters
        ---------
        bitstring
            Initial bitstring for the (data) qubits

        Returns
        -------
        cov
            Covariance matrix of the initialized FGS
        norm
            Norm of the initialized FGS

        References
        ----------
        [1] Suzuki, Yasunari, Keisuke Fujii, and Masato Koashi. "Efficient simulation of quantum
        error correction under coherent error based on the nonunitary free-fermionic formalism."
        Physical review letters 119.19 (2017): 190503.

        [2] Bravyi, Sergey. "Lagrangian representation for fermionic linear optics."
        arXiv preprint quant-ph/0404180 (2004).
        """
        norm = 1
        cov = np.zeros((2 * (self.num_data + 1), 2 * (self.num_data + 1)))

        # Warning: for now, we have only implemented it for b=0,...,0
        assert not (np.array(list(bitstring.values()))).any()

        cov[0, -1] = -1
        cov[-1, 0] = +1
        for i in range(self.num_data):
            cov[2 * i + 1, 2 * i + 2] = -1
            cov[2 * i + 2, 2 * i + 1] = +1

        return cov, norm

    def log_meas(self, state: Tuple[np.ndarray, float], eps=1e-6) -> float:
        """
        Returns the posterior probability of having a logical errors given an
        (unnormalized) FGS, meaning the probability that the initial logical
        state is not the same as the measured one given the observed syndromes.

        The used expression is Eq. S4 from Ref. [1].

        Parameters
        ----------
        state
            FGS before the logical measurement.
        eps
            Tolerance for numerical instabilities when calculating the probability.
            Default: 1e-6.

        Returns
        -------
        log_error_posterior
            Posterior probability of having a logical error.

        References
        ----------
        [1] Suzuki, Yasunari, Keisuke Fujii, and Masato Koashi. "Efficient simulation of quantum
        error correction under coherent error based on the nonunitary free-fermionic formalism."
        Physical review letters 119.19 (2017): 190503.
        """
        cov, norm = state

        log_error_posterior = 0.5 * (
            1 - cov[2 * (self.num_data + 1) - 2, 2 * (self.num_data + 1) - 3]
        )

        # numerical issue where `prob = 1.00000000001.. > 1` or negative
        if log_error_posterior > 1 and log_error_posterior < 1 + eps:
            log_error_posterior = 1
        if log_error_posterior < 0 and log_error_posterior > -eps:
            log_error_posterior = 0

        return log_error_posterior

    def final_qec_round(
        self, state: Tuple[np.ndarray, float]
    ) -> Tuple[Tuple[np.ndarray, float], np.ndarray]:
        """
        Returns the state and measured syndrome from an ideal final QEC round given an FGS.
        The algorithm is based on Ref. [1].

        Paramters
        ---------
        state
            Unnormalized FGS before the final QEC round.

        Returns
        -------
        out_state
            Unnormalized FGS after the final QEC round.
        synd
            Syndrome outcome vector measured in the final QEC round.

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

            state, synd, self._pre_kraus = apply_ideal_parity(
                state=state,
                qubit_ids=qubit_ids,
                num_data=self.num_data,
                pre_kraus=self._pre_kraus,
            )
            synd_vector.append(synd)

        return state, np.array(synd_vector)

    def qec_round(
        self, state: Tuple[np.ndarray, float]
    ) -> Tuple[Tuple[np.ndarray, float], np.ndarray]:
        """
        Returns the state and measured syndrome from an ideal QEC round given an FGS.
        The algorithm is based on Ref. [1].

        Paramters
        ---------
        state
            FGS before the QEC round.

        Returns
        -------
        out_state
            FGS after the QEC round.
        synd
            Syndrome outcome vector measured in the QEC round.

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

            state, synd, self._pre_kraus = apply_ideal_parity(
                state=state,
                qubit_ids=qubit_ids,
                num_data=self.num_data,
                pre_kraus=self._pre_kraus,
            )
            synd_vector.append(synd)

        return state, np.array(synd_vector)

    def idle(self, state: Tuple[np.ndarray, float]) -> Tuple[np.ndarray, float]:
        """
        Returns the state after ideal idling (i.e. doing nothing).

        Paramters
        ---------
        state
            FGS before the idling.

        Returns
        -------
        out_state
            FGS after the idling.
        """
        return state
