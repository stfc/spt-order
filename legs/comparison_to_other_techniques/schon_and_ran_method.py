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


import copy
import json
import os
import pickle
from adaptaqc.utils.utilityfunctions import qiskit_to_tenpy_mps
from mps_to_circuit import mps_to_circuit
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_addon_aqc_tensor.simulation import tensornetwork_from_circuit
from qiskit.qasm3 import dumps
from tenpy.networks.mps import MPS
from typing import Dict


def compare_circuit_and_mps(circuit: QuantumCircuit, mps: MPS) -> Dict:
    """
    Takes a QuantumCircuit and a TeNPy MPS and returns a dictionary containing their fidelity and
    some properties of the circuit.

    Args:
        circuit (QuantumCircuit): A Qiskit circuit.
        mps (MPS): A TeNPy MPS.
    Returns:
        (Dict): Dictionary of fidelity and circuit properties.
    """

    circuit_mps = qiskit_to_tenpy_mps(
        tensornetwork_from_circuit(
            circuit,
            AerSimulator(
                method="matrix_product_state",
                matrix_product_state_truncation_threshold=1e-16,
                matrix_product_state_max_bond_dimension=100,
            ),
        )._as_tuple()
    )

    # Check if the transpiler permuted the qubits, and if so, permute back.
    layout = circuit.layout.final_index_layout()

    if not (layout == [i for i in range(l)]):
        print("Indices have been permuted. Permuting back...")

        perm = [layout.index(i) for i in range(l)]
        circuit_mps.permute_sites(perm)

    # Calculate metrics
    compiled_fidelity = abs(mps.overlap(circuit_mps)) ** 2
    ops = circuit.count_ops()
    depth = circuit.depth()
    cx_depth = circuit.depth(filter_function=lambda instr: len(instr.qubits) > 1)

    print(f"Fidelity with original target: {compiled_fidelity}")
    print(f"Ops: {ops}")
    print(f"Depth: {depth}")
    print(f"CX depth: {cx_depth}")

    return {
        "circuit_fidelity_with_original_target": compiled_fidelity,
        "ops": ops,
        "depth": depth,
        "cx_depth": cx_depth,
        "circuit": dumps(circuit),
    }


l = 100
method = "ran"  # Choose "schon" or "ran"
num_layers = 10  # Only relevant for Ran method
# Set to None to not compress the target. Set to X to compress to the minimum bond dimension with
# at least X fidelity with the target.
compress_to_at_least = None
result_dir = "results_bahm/"
Path(result_dir).mkdir(parents=True, exist_ok=True)

mps_fn = "../bond_alternating_heisenberg_model/saved_mps/target_n_100_J0_0.5_J1_1.0_hz_0.0_trunc_1e-12_compressed_bd_None.pkl"
with open(mps_fn, "rb") as mps_f:
    raw_target_mps = pickle.load(mps_f)

mps = qiskit_to_tenpy_mps(raw_target_mps)
print(max(mps.chi))
original_target_mps = copy.deepcopy(mps)

if compress_to_at_least is None:
    compressed_target_fidelity = 1
    compressed_target_mps = mps
else:
    # Compress to max bond dimension 1, 2, 3, ... until `compress_to_at_least` fidelity is reached.
    for max_bond in range(1, max(original_target_mps.chi) + 1):

        mps = qiskit_to_tenpy_mps(raw_target_mps)

        compression_options = {
            "compression_method": "variational",
            "trunc_params": {"chi_max": max_bond},
            "max_trunc_err": 1,
            "max_sweeps": 50,
            "min_sweeps": 10,
        }
        mps.compress(compression_options)
        mps.norm = 1

        fidelity = abs(mps.overlap(original_target_mps)) ** 2
        print(f"Max bond: {max(mps.chi)}, fidelity: {fidelity}")

        if fidelity >= compress_to_at_least:
            compressed_target_fidelity = fidelity
            compressed_target_mps = mps
            break

print(compressed_target_mps.sites[0].get_op("Sz").to_ndarray())

# Extract tensors from MPS.
# We flip the order of the physical dimension due to the convention.
mps_arrays = [
    compressed_target_mps.get_B(i, form="A")
    .itranspose(["vL", "p", "vR"])
    .to_ndarray()[:, ::-1, :]
    for i in range(l)
]

results = {
    "method": method,
    "original_target_mps_fn": mps_fn,
    "original_target_mps_chi": max(original_target_mps.chi),
    "compress_to_at_least": compress_to_at_least,
    "compressed_target_mps_fidelity": compressed_target_fidelity,
    "compressed_target_mps_chi": max(compressed_target_mps.chi),
}

if method == "schon":
    # Schon method
    qc = mps_to_circuit(mps_arrays, method="exact", shape="lpr")

    # Transpile to a linear coupling map. We transpile twice because transpiling to this basis at the
    # same time as to a linear coupling map throws an error.
    coupling_map = CouplingMap.from_line(l)
    transpiled_qc = transpile(
        qc,
        basis_gates=["rx", "ry", "rz", "cx"],
        optimization_level=2,
        layout_method="trivial",
    )
    transpiled_qc = transpile(
        transpiled_qc,
        basis_gates=["rx", "ry", "rz", "cx"],
        coupling_map=coupling_map,
        optimization_level=2,
        layout_method="trivial",
    )

    metrics = compare_circuit_and_mps(transpiled_qc, original_target_mps)
    results = results | metrics

elif method == "ran":
    # Ran method
    coupling_map = CouplingMap.from_line(l)
    circuits = {"circuits": []}
    _ = mps_to_circuit(
        mps_arrays,
        method="approximate",
        shape="lpr",
        num_layers=num_layers,
        history=circuits,
    )

    for layers in range(1, num_layers + 1):
        qc = circuits["circuits"][layers - 1]
        transpiled_qc = transpile(
            qc,
            basis_gates=["rx", "ry", "rz", "cx"],
            coupling_map=coupling_map,
            optimization_level=2,
            layout_method="trivial",
        )

        metrics = compare_circuit_and_mps(transpiled_qc, original_target_mps)
        results[layers] = {"layers": layers} | metrics

else:
    raise ValueError(f"Invalid method: {method}. Choose from 'schon' or 'ran'")

result_fn = f"{method}_method_compress_to_{compress_to_at_least}_{mps_fn.split('/')[-1].replace('pkl', 'json')}"
with open(os.path.join(result_dir, result_fn), "w") as file:
    json.dump(results, file, indent=4)
