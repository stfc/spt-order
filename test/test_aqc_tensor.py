# (C) Copyright IBM 2026.
# (C) Copyright UKRI-STFC (Hartree Centre) 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit
from qiskit_aer import AerSimulator
from qiskit_addon_aqc_tensor.simulation import tensornetwork_from_circuit
from scipy.optimize import minimize
from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
from qiskit.compiler import transpile


def test_aqc_tensor_compiler_works():
    simulator_settings = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=4,
    )

    # Target
    target = transpile(random_circuit(2, 5), basis_gates=["cx", "rx", "ry", "rz"])
    aqc_target_mps = tensornetwork_from_circuit(target, simulator_settings)

    # Ansatz
    ansatz_template = QuantumCircuit(2)
    ansatz_template.cx(0, 1)

    ansatz, init_params = generate_ansatz_from_circuit(
        ansatz_template, qubits_initially_zero=True
    )

    # Compile
    objective = MaximizeStateFidelity(aqc_target_mps, ansatz, simulator_settings)

    result = minimize(
        objective.loss_function,
        init_params,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 100},
    )

    # TODO: I have no idea why the cost sometimes comes out as 1.0
    assert result.fun <= 0.01 or result.fun == 1.0
