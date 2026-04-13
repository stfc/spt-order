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


import numpy as np
from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector, state_fidelity

# IBM Kingston basis gates:
basis_gates = ["cz", "id", "rx", "rz", "rzz", "sx", "x"]

# 1. Single two-qubit identity block transpilation.

# Two-qubit identity operator
I = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)

# Two-qubit zero state.
zero_state = np.zeros(4)
zero_state[0] = 1

qc = QuantumCircuit(2)

qc.unitary(I, [0, 1])
print(qc)

aqc_ansatz, aqc_initial_parameters = generate_ansatz_from_circuit(
    qc, qubits_initially_zero=True
)

assigned_ansatz = aqc_ansatz.assign_parameters(aqc_initial_parameters)

# Transpile
for optimization_level in range(4):
    transpiled_ansatz = transpile(
        assigned_ansatz,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
    )

    print(f"optimization_level={optimization_level}")
    print("Circuit:")
    print(transpiled_ansatz)
    print(f"Ops: {transpiled_ansatz.count_ops()}")
    print(
        f"CX depth: {transpiled_ansatz.depth(filter_function=lambda instr: len(instr.qubits) > 1)}"
    )
    print(
        f"State fidelity with |0...0>: {state_fidelity(Statevector(zero_state), Statevector(transpiled_ansatz))}"
    )
    print("")


# 2. Brickwork of two-qubit identity blocks transpilation:
n = 10
layers = 4
qc_brickwork = QuantumCircuit(n)

# Add two-qubit identities in a brickwall pattern.
for _ in range(layers):
    for i in range(0, n - 1, 2):
        qc_brickwork.unitary(I, [i, i + 1])
    for i in range(1, n - 1, 2):
        qc_brickwork.unitary(I, [i, i + 1])
print(qc_brickwork)

aqc_brickwork_ansatz, aqc_brickwork_initial_parameters = generate_ansatz_from_circuit(
    qc_brickwork, qubits_initially_zero=True
)

assigned_brickwork_ansatz = aqc_brickwork_ansatz.assign_parameters(
    aqc_brickwork_initial_parameters
)

# n-qubit zero state.
zero_state = np.zeros(2**n)
zero_state[0] = 1

# Transpile
for optimization_level in range(4):
    transpiled_ansatz = transpile(
        assigned_brickwork_ansatz,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
    )

    print(f"optimization_level={optimization_level}")
    print("Circuit:")
    print(transpiled_ansatz)
    print(f"Ops: {transpiled_ansatz.count_ops()}")
    print(
        f"CX depth: {transpiled_ansatz.depth(filter_function=lambda instr: len(instr.qubits) > 1)}"
    )
    print(
        f"State fidelity with |0...0>: {state_fidelity(Statevector(zero_state), Statevector(transpiled_ansatz))}"
    )
    print("")


# NOTE: optimization_level>1 causes the circuit to reduce to contain no two-qubit gates. We should
# use optimization_level=1.
