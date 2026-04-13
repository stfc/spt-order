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


import json
import os
import pickle
from adaptaqc.utils.utilityfunctions import qiskit_to_tenpy_mps
from pathlib import Path
from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_addon_aqc_tensor.simulation import tensornetwork_from_circuit
from qiskit.qasm3 import loads

l = 100

# Load target MPS.
mps_fn = "../bond_alternating_heisenberg_model_compiling/saved_mps/target_n_100_J0_1.0_J1_-2.0_hz_0.0_trunc_1e-12_compressed_bd_None.pkl"
with open(mps_fn, "rb") as mps_f:
    raw_target_mps = pickle.load(mps_f)

target_mps = qiskit_to_tenpy_mps(raw_target_mps)

results = {
    "method": "AQC",
    "target_mps_fn": mps_fn,
}

# Load compiled circuits. NOTE: you can add multiple compiled circuits for a given target MPS.
# Each one will go as a separate entry.
circuit_fns = [
    "../bond_alternating_heisenberg_model_compiling/results/n_100_layers_6_J0_1.0_J1_-2.0_tt_1e-12_1756388702662351325_fidelity_0.9794.txt"
]

for index in range(len(circuit_fns)):
    circuit_fn = circuit_fns[index]
    with open(circuit_fn, "r") as circuit_f:
        qasm_string = circuit_f.read()
        compiled_circuit = loads(qasm_string)

    # Transpile to a linear coupling map.
    coupling_map = CouplingMap.from_line(l)
    transpiled_qc = transpile(
        compiled_circuit,
        basis_gates=["rx", "ry", "rz", "cx"],
        coupling_map=coupling_map,
        optimization_level=2,
        layout_method="trivial",
    )

    # Compute fidelity.
    circuit_mps = qiskit_to_tenpy_mps(
        tensornetwork_from_circuit(
            transpiled_qc,
            AerSimulator(
                method="matrix_product_state",
                matrix_product_state_truncation_threshold=1e-16,
                matrix_product_state_max_bond_dimension=100,
            ),
        )._as_tuple()
    )

    # Check if the transpiler permuted the qubits, and if so, permute back.
    layout = transpiled_qc.layout.final_index_layout()

    if not (layout == [i for i in range(l)]):
        print("Indices have been permuted. Permuting back...")

        perm = [layout.index(i) for i in range(l)]
        circuit_mps.permute_sites(perm)

    # Calculate metrics
    fidelity = abs(target_mps.overlap(circuit_mps)) ** 2
    ops = transpiled_qc.count_ops()
    depth = transpiled_qc.depth()
    cx_depth = transpiled_qc.depth(filter_function=lambda instr: len(instr.qubits) > 1)

    print(f"Index: {index}")
    print(f"Fidelity: {fidelity}")
    print(f"Ops: {ops}")
    print(f"Depth: {depth}")
    print(f"CX depth: {cx_depth}")

    results[index] = {
        "compiled_circuit_fn": circuit_fn,
        "fidelity": fidelity,
        "ops": ops,
        "depth": depth,
        "cx_depth": cx_depth,
    }

result_fn = f"aqc_method_{mps_fn.split('/')[-1].replace('pkl', 'json')}"
result_dir = "results_bahm/"
Path(result_dir).mkdir(parents=True, exist_ok=True)
with open(os.path.join(result_dir, result_fn), "w") as file:
    json.dump(results, file, indent=4)
