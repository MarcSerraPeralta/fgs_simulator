from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, Union, List

import yaml

T = TypeVar("T", bound="Setup")


class Setup:
    def __init__(self, setup: Dict[str, Any]) -> None:
        self._qubit_params = dict()
        self._global_params = dict()

        _setup = deepcopy(setup)
        self.name = _setup.pop("name", "")
        self.description = _setup.pop("description", "")
        self.load_params(_setup.pop("setup"))
        return

    def load_params(self, params: Dict[str, Any]) -> None:
        if not params:
            raise ValueError("setup not found or contains no information")

        for params_dict in params:
            if "qubit" in params_dict:
                qubit = str(params_dict.pop("qubit"))
                qubits = (qubit,)
            elif "qubits" in params_dict:
                qubits = tuple(params_dict.pop("qubits"))
            else:
                qubits = None

            if qubits:
                if qubits in self._qubit_params.keys():
                    raise ValueError("Parameters defined repeatedly in the setup.")
                self._qubit_params[qubits] = params_dict
            else:
                self._global_params.update(params_dict)

    @classmethod
    def from_yaml(cls: Type[T], filename: Union[str, Path]) -> T:
        """
        Create new fgs_sim.Setup instance from YAML configuarion file.

        Parameters
        ----------
        filename
            The YAML file name.

        Returns
        -------
        T
            The initialised fgs_sim.Setup object based on the yaml.
        """
        with open(filename, "r") as file:
            setup = yaml.safe_load(file)
            return cls(setup)

    def to_dict(self) -> Dict[str, Any]:
        setup = dict()

        setup["name"] = self.name
        setup["description"] = self.description

        qubit_params = []
        if self._global_params:
            qubit_params.append(self._global_params)

        for qubits, params in self._qubit_params.items():
            params_copy = deepcopy(params)
            num_qubits = len(qubits)
            if num_qubits == 1:
                params_copy["qubit"] = qubits[0]
            elif num_qubits == 2:
                params_copy["qubits"] = list(qubits)
            qubit_params.append(params_copy)

        setup["setup"] = qubit_params

        return setup

    def to_yaml(self, filename: Union[str, Path]) -> None:
        setup = self.to_dict()
        with open(filename, "w") as file:
            yaml.dump(setup, file, default_flow_style=False)
        return

    def set_param(self, param: str, param_val: float, *qubits: str) -> None:
        if not qubits:
            self._global_params[param] = param_val
        else:
            self._qubit_params[qubits][param] = param_val

    def param(self, param: str, *qubits: str) -> float:
        try:
            val = self._qubit_params[qubits][param]
            return self._eval_param_val(val)
        except KeyError:
            pass

        try:
            val = self._global_params[param]
            return val
        except KeyError:
            pass

        if qubits:
            qubit_str = ", ".join(qubits)
            raise KeyError(f"Parameter {param} not defined for qubit(s) {qubit_str}")
        raise KeyError(f"Global parameter {param} not defined")
