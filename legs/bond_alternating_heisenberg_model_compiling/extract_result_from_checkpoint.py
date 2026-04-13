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


import pickle
import os
import numpy as np
import quimb.tensor as qtn
from functools import partial
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit
from qiskit_addon_aqc_tensor.simulation.aer import QiskitAerMPS
from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator
from qiskit_aer import AerSimulator
from adaptaqc.utils.utilityfunctions import qiskit_to_tenpy_mps
from qiskit_addon_aqc_tensor.simulation import (
    compute_overlap,
    tensornetwork_from_circuit,
)

checkpoint_fn = "./checkpoint_J1_-1.0/n_100_layers_2_J0_1.0_J1_-1.0_tt_1e-12_1756388302044410333/chkpt_619969.pkl"

# Reconstruct timestamp when job was created
TIME = checkpoint_fn.split("_")[-2].split("/")[0]

with open(checkpoint_fn, "rb") as chkpt_f:
    checkpoint = pickle.load(chkpt_f)

original_args = checkpoint["args"]
intermediate_result = checkpoint["intermediate_result"]


# --------------------------------------------------------------------------------
# Helper funcations
def get_backend():
    if original_args["backend"] == "MPS_SIM":
        return AerSimulator(
            method="matrix_product_state",
            matrix_product_state_truncation_threshold=original_args["mps_truncation"],
            matrix_product_state_max_bond_dimension=original_args[
                "mps_max_bond_dimension"
            ],
        )
    elif original_args["backend"] == "QUIMB":
        return QuimbSimulator(
            partial(
                qtn.CircuitMPS,
                gate_opts={
                    "cutoff": original_args["mps_truncation"],
                    "max_bond": original_args["mps_max_bond_dimension"],
                },
            ),
            autodiff_backend="jax",
        )
    else:
        raise ValueError("Unrecognized backend")


n = original_args["qubits"]
J0 = original_args["even_coupling"]
J1 = original_args["odd_coupling"]
target_trunc = original_args["target_trunc"]
layers = original_args["layers"]
filename_parts = [
    f"n_{n}",
    f"layers_{layers}",
    f"J0_{J0}",
    f"J1_{J1}",
    f"tt_{target_trunc}",
    f"{TIME}",
]
filename = "_".join(filename_parts)


# Reconstruct the ansatz
target_dir = "saved_mps/"
target_fn = f"target_n_{n}_J0_{J0}_J1_{J1}_hz_0.0_trunc_{target_trunc}_compressed_bd_{original_args["compressed_bond_dim"] if original_args["compressed_bond_dim"] > 0 else None}.pkl"
with open(target_dir + target_fn, "rb") as mps_f:
    raw_target_mps = pickle.load(mps_f)


qc = QuantumCircuit(n)
for _ in range(layers):
    for i in range(0, n - 1, 2):
        qc.cx(i, i + 1)
    for i in range(1, n - 1, 2):
        qc.cx(i, i + 1)

# Initialise as a product of Bell pairs: |01> - |10>
for i in range(0, n - 1, 2):
    qc.x([i, i + 1])
    qc.h(i)
    qc.cx(i, i + 1)

aqc_ansatz, _ = generate_ansatz_from_circuit(qc, qubits_initially_zero=True)

# Initialise with the checkpoint params
intermediate_params = intermediate_result.x


fidelity = (
    abs(
        compute_overlap(
            tensornetwork_from_circuit(
                transpile(
                    aqc_ansatz.assign_parameters(intermediate_params),
                    basis_gates=["cx", "rx", "ry", "rz"],
                ),
                AerSimulator(
                    method="matrix_product_state",
                    matrix_product_state_truncation_threshold=1e-16,
                ),
            ),
            QiskitAerMPS(raw_target_mps[0], raw_target_mps[1]),
        )
    )
    ** 2
)

print(f"Fidelity: {fidelity}")

backend = get_backend()

# Get the target MPS in the correct format.
if isinstance(backend, AerSimulator):
    target_mps: QiskitAerMPS = QiskitAerMPS(raw_target_mps[0], raw_target_mps[1])
elif isinstance(backend, QuimbSimulator):
    # Raw MPS to Tenpy
    tenpy_mps = qiskit_to_tenpy_mps(raw_target_mps, return_form="SpinHalfSite")
    # Convert Tenpy MPS to Quimb
    arrays = [
        tenpy_mps.get_B(i, form="A").itranspose(["vL", "vR", "p"]).to_ndarray()
        for i in range(n)
    ]

    arrays[0] = np.squeeze(arrays[0], axis=0)
    arrays[-1] = np.squeeze(arrays[-1], axis=1)

    quimb_mps = qtn.tensor_1d.MatrixProductState(arrays=arrays, shape="lrp")
    quimb_mps.left_canonize()

    target_mps = quimb_mps


# Save result

result_dir = "results/"
Path(result_dir).mkdir(parents=True, exist_ok=True)
result_data = {
    "args": original_args,
    "result": intermediate_result,
    "target_mps": target_mps,
    "aqc_ansatz": aqc_ansatz,
}

with open(os.path.join(result_dir, filename) + ".pkl", mode="wb") as file:
    pickle.dump(result_data, file)
