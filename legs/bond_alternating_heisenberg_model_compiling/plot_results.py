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


import logging
import pickle

from qiskit import transpile
from qiskit.qasm3 import dumps
from qiskit_addon_aqc_tensor.simulation.aer import (
    QiskitAerMPS,
    tensornetwork_from_circuit,
    compute_overlap,
)
from qiskit_aer import AerSimulator
from adaptaqc.utils.utilityfunctions import qiskit_to_tenpy_mps
from adaptaqc.utils.entanglement_measures import concurrence
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Params
n = 100
J0 = 1.0
J1 = 0.5
target_trunc = 1e-12
compressed_bond_dim = 5

result_fn = "results/n_100_layers_3_J0_1.0_J1_0.5_tt_1e-12_1756375695282178000.pkl"

# Target
target_fn = f"target_n_{n}_J0_{J0}_J1_{J1}_hz_0.0_trunc_{target_trunc}_compressed_bd_{compressed_bond_dim}.pkl"
with open("saved_mps/" + target_fn, "rb") as mps_f:
    raw_target_mps = pickle.load(mps_f)

target_mps: QiskitAerMPS = QiskitAerMPS(raw_target_mps[0], raw_target_mps[1])

logger.info(
    f"Loaded MPS of max bond dimension: {max([len(a) for a in target_mps.lamb])}"
)

# Result
with open(result_fn, "rb") as aqc_f:
    data = pickle.load(aqc_f)

param_vals = data["result"].x
aqc_ansatz = data["aqc_ansatz"]
trained_qc = aqc_ansatz.assign_parameters(param_vals)
transpiled_qc = transpile(
    trained_qc, basis_gates=["cx", "rx", "ry", "rz"], optimization_level=0
)


simulator_settings = AerSimulator(
    method="matrix_product_state",
    matrix_product_state_truncation_threshold=1e-16,
)

fidelity = (
    abs(
        compute_overlap(
            tensornetwork_from_circuit(transpiled_qc, simulator_settings), target_mps
        )
    )
    ** 2
)

# Visualise
Path("figures/").mkdir(parents=True, exist_ok=True)
fn = f"figures/n_{n}_J0_{J0}_J1_{J1}_tt_{target_trunc}_fidelity_{str(fidelity)[:6]}"

# Circuit
transpiled_qc.draw(output="mpl", fold=-1)
plt.savefig(fn + "_circuit.png", dpi=100)

# Save circuit as QASM
qasm_string = dumps(transpiled_qc)
new_fn = result_fn[:-4]
with open(f"{new_fn}_fidelity_{str(fidelity)[:6]}.txt", "w") as file:
    file.write(qasm_string)

tenpy_mps_compiled = qiskit_to_tenpy_mps(
    tensornetwork_from_circuit(transpiled_qc, simulator_settings)._as_tuple()
)
tenpy_mps_original = qiskit_to_tenpy_mps(target_mps._as_tuple())

# Sz
fig1, ax1 = plt.subplots()
ax1.plot(tenpy_mps_original.expectation_value("Sz"), label="Original", marker="x")
ax1.plot(tenpy_mps_compiled.expectation_value("Sz"), label="Compiled", marker="x")
ax1.set_ylim(-0.6, 0.6)
ax1.set_ylabel(r"$\langle S_z \rangle$")
ax1.set_xlabel("Site")
ax1.set_title(f"Fidelity: {str(fidelity)[:6]}")
ax1.legend()
fig1.tight_layout()
fig1.savefig(fn + "_sz.png", dpi=100)

# Bond dimension
fig, axs = plt.subplots(1, 2)
axs[0].plot(tenpy_mps_original.entanglement_entropy(), label="Original")
axs[0].plot(tenpy_mps_compiled.entanglement_entropy(), label="Compiled")
axs[0].set_ylabel("Entanglement entropy S")
axs[0].set_xlabel("Bond")
axs[0].set_ylim(bottom=0)
axs[1].plot(tenpy_mps_original.chi, label="Original")
axs[1].plot(tenpy_mps_compiled.chi, label="Compiled")
axs[1].set_ylabel(r"Bond dimension $\chi$")
axs[1].set_xlabel("Bond")
axs[1].set_ylim(0, 50)
axs[0].legend()
axs[1].legend()
plt.suptitle(f"Fidelity: {str(fidelity)[:6]}")
fig.tight_layout()
plt.savefig(fn + "_bond_dim.png", dpi=100)


# Concurrence
fig2, ax2 = plt.subplots()
rhos_compiled = [
    tenpy_mps_compiled.get_rho_segment([i, i + 1]).to_ndarray().reshape(4, 4)
    for i in range(n - 1)
]
rhos_original = [
    tenpy_mps_original.get_rho_segment([i, i + 1]).to_ndarray().reshape(4, 4)
    for i in range(n - 1)
]
ax2.plot([concurrence(rho) for rho in rhos_original], label="Original")
ax2.plot([concurrence(rho) for rho in rhos_compiled], label="Compiled")
ax2.set_ylim(0, 1)
ax2.set_ylabel("Nearest-neighbour concurrence")
ax2.set_xlabel("Site")
ax2.legend()
ax2.set_title(f"Fidelity: {str(fidelity)[:6]}")
fig2.tight_layout()
fig2.savefig(fn + "_concurrence.png", dpi=100)
