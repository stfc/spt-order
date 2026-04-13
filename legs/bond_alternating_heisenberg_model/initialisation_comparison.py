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
import numpy as np
import quimb.tensor as qtn
from functools import partial
from adaptaqc.utils.utilityfunctions import qiskit_to_tenpy_mps
from legs.optimizers import adam
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator
from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit
from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
from qiskit_addon_aqc_tensor.simulation import (
    compute_overlap,
    tensornetwork_from_circuit,
)
from qiskit_addon_aqc_tensor.simulation.aer import QiskitAerMPS
from scipy.optimize import OptimizeResult, minimize

# Load MPS
target_fn = (
    "saved_mps/target_n_100_J0_0.5_J1_1.0_hz_0.0_trunc_1e-12_compressed_bd_5.pkl"
)
with open(target_fn, "rb") as mps_f:
    raw_target_mps = pickle.load(mps_f)

reference_mps = qiskit_to_tenpy_mps(raw_target_mps)

# 1. Product state
# Use TeNPy to compress to χ=1.
print("----------------------------------------------------------------")
print("----------------------- 1. Product state -----------------------")
print("----------------------------------------------------------------")
product_mps = qiskit_to_tenpy_mps(raw_target_mps)

compression_options = {
    "compression_method": "variational",
    "trunc_params": {"chi_max": 1},
    "max_trunc_err": 1,
    "max_sweeps": 50,
    "min_sweeps": 10,
}
product_mps.compress(compression_options)
product_mps.norm = 1

product_fidelity = abs(product_mps.overlap(reference_mps)) ** 2
print(f"Max bond: {max(product_mps.chi)}, fidelity: {product_fidelity}")
print("")


# 2. chi=2 MPS
# Use TeNPy to compress to χ=2 (absolute upper bound for the following methods).
print("----------------------------------------------------------------")
print("------------------------- 2. chi=2 MPS -------------------------")
print("----------------------------------------------------------------")
chi_2_mps = qiskit_to_tenpy_mps(raw_target_mps)

compression_options = {
    "compression_method": "variational",
    "trunc_params": {"chi_max": 2},
    "max_trunc_err": 1,
    "max_sweeps": 50,
    "min_sweeps": 10,
}
chi_2_mps.compress(compression_options)
chi_2_mps.norm = 1

chi_2_fidelity = abs(chi_2_mps.overlap(reference_mps)) ** 2
print(f"Max bond: {max(chi_2_mps.chi)}, fidelity: {chi_2_fidelity}")
print("")


# 3. Alternating 2-1 bond dimension: χ=[2, 1, 2, 1, ...]
# Use TeNPy to compress to χ=[2, 1, 2, 1, ...]
print("----------------------------------------------------------------")
print("--- 3. Alternating 2-1 bond dimension: chi=[2, 1, 2, 1, ...] ---")
print("----------------------------------------------------------------")
alternating_chi_mps = qiskit_to_tenpy_mps(raw_target_mps)

print(
    f"Original chi: {alternating_chi_mps.chi[:6]}... len={len(alternating_chi_mps.chi)}"
)

# Group the sites: [0,1]-[2,3]-[4,5]-...
alternating_chi_mps.group_sites(2)
print(
    f"Grouped chi: {alternating_chi_mps.chi[:6]}... len={len(alternating_chi_mps.chi)}"
)

# Compress the bonds between groups to χ=1
# [0,1]-(χ=1)-[2,3]-(χ=1)-[4,5]-...
compression_options = {
    "compression_method": "variational",
    "trunc_params": {"chi_max": 1},
    "max_trunc_err": 1,
    "max_sweeps": 50,
    "min_sweeps": 10,
}
alternating_chi_mps.compress(compression_options)
alternating_chi_mps.norm = 1
print(
    f"Compressed grouped chi: {alternating_chi_mps.chi[:6]}... len={len(alternating_chi_mps.chi)}"
)

# Un-group the sites, with a max in-group χ=2 (I think this is the maximum possible anyway)
# [0]-(χ=2)-[1]-(χ=1)-[2]-(χ=2)-[3]-(χ=1)-...
alternating_chi_mps.group_split(trunc_par={"chi_max": 2})
print(
    f"Compressed un-grouped chi: {alternating_chi_mps.chi[:6]}... len={len(alternating_chi_mps.chi)}"
)


bell_like_fidelity = abs(alternating_chi_mps.overlap(reference_mps)) ** 2
print(f"Alternating chi=[2, 1, 2, 1, ...] fidelity: {bell_like_fidelity}")
print("")


# 4. Explicit Bell pairs: |01> - |10>
# Use a QuantumCircuit to explicitly construct the Bell pair state.
print("----------------------------------------------------------------")
print("------------- 4. Explicit Bell pairs: |01> - |10> --------------")
print("----------------------------------------------------------------")
qc = QuantumCircuit(100)

# Product of Bell pairs: |01> - |10>. Change 0-99 to 1-98 for the odd-Haldane phase.
for i in range(0, 99, 2):
    qc.x([i, i + 1])
    qc.h(i)
    qc.cx(i, i + 1)

bell_fidelity = (
    abs(
        compute_overlap(
            tensornetwork_from_circuit(
                qc,
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

print(f"Bell pairs: {bell_fidelity}")
print("")


# 5. Optimised Bell pair circuit
# Take the previous circuit, kick its parameters slightly away from the Bell state, and use
# AQC-Tensor to optimise the parameters.
print("----------------------------------------------------------------")
print("---------------- 5. Optimised Bell pair circuit ----------------")
print("----------------------------------------------------------------")

aqc_ansatz, aqc_initial_parameters = generate_ansatz_from_circuit(
    qc, qubits_initially_zero=True
)

# Try to kick out of Bell state potential local minimum
# Each parameter is chosen randomly within (val-tol) and (val+tol), val: original value.
tol = 0.2
for i in range(len(aqc_initial_parameters)):
    aqc_initial_parameters[i] = aqc_initial_parameters[i] + (
        (np.random.random() - 0.5) * 2 * tol
    )


backend = QuimbSimulator(
    partial(qtn.CircuitMPS, gate_opts={"cutoff": 1e-8, "max_bond": 100}),
    autodiff_backend="jax",
)

# Raw MPS to Tenpy
tenpy_mps = qiskit_to_tenpy_mps(raw_target_mps, return_form="SpinHalfSite")
# Convert Tenpy MPS to Quimb
arrays = [
    tenpy_mps.get_B(i, form="A").itranspose(["vL", "vR", "p"]).to_ndarray()
    for i in range(100)
]

arrays[0] = np.squeeze(arrays[0], axis=0)
arrays[-1] = np.squeeze(arrays[-1], axis=1)

quimb_mps = qtn.tensor_1d.MatrixProductState(arrays=arrays, shape="lrp")
quimb_mps.left_canonize()

target_mps = quimb_mps

objective = MaximizeStateFidelity(target_mps, aqc_ansatz, backend)


def callback(intermediate_result: OptimizeResult):
    nit = intermediate_result.nit

    if nit == 0:
        print(f"Initial fidelity: {1 - intermediate_result.fun}")


result = minimize(
    objective.loss_function,
    aqc_initial_parameters,
    method=adam,
    jac=True,
    options={
        "maxiter": int(1e3),
        "ftol": np.nan,
        "gtol": np.sqrt(np.finfo(float).tiny),
    },
    callback=callback,
)

print(f"Final fidelity: {1 - result.fun}, nit: {result.nit}")
