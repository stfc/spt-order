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
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import time
from pathlib import Path
from qiskit.qasm3 import loads
from qiskit_aer import AerSimulator
from qiskit_addon_aqc_tensor.simulation import tensornetwork_from_circuit
from adaptaqc.utils.utilityfunctions import qiskit_to_tenpy_mps
from tenpy import MPS

plt.rc("font", family="serif")
plt.rcParams.update({"font.size": 12})
plt.rc("text", usetex=True)


def correlation_function(mps: MPS):
    correlators = np.empty((mps.L, mps.L))
    correlators[:] = np.nan
    z = mps.expectation_value("Sz")
    for i in range(mps.L):
        for j in range(i + 1, min(i + 20, mps.L - 1) + 1):
            zizj = mps.expectation_value_term([("Sz", i), ("Sz", j)])
            correlator = zizj - z[i] * z[j]

            correlators[i, j] = correlator
            correlators[j, i] = correlator

    return correlators


n = 100
J0 = 1.0
J1 = -2.0

plot_dir = (
    f"./figures/correlation_comparison/n_{n}_J0_{J0}_J1_{J1}_{str(time.time_ns())[:-9]}"
)
Path(plot_dir).mkdir(parents=True, exist_ok=True)

# Specify the sites i for which to plot Czz(i, j) as individual plots
specific_indices = [0, 50]

# Add file-paths for the target MPS, an AQC-compiled circuit, and the Ran method metrics JSON.
files = {
    "target": "../bond_alternating_heisenberg_model/saved_mps/target_n_100_J0_1.0_J1_-2.0_hz_0.0_trunc_1e-12_compressed_bd_8.pkl",
    "AQC": "../bond_alternating_heisenberg_model_compiling/results/n_100_layers_6_J0_1.0_J1_-2.0_tt_1e-12_1756388702662351325_fidelity_0.9794.txt",
    "Ran": "../comparison_to_other_techniques/results_bahm/ran_method_compress_to_None_target_n_100_J0_1.0_J1_-2.0_hz_0.0_trunc_1e-12_compressed_bd_8.json",
}

# Save the file paths in the same directory, so it's clear which MPSs were used.
with open(os.path.join(plot_dir, "filepaths.json"), "w") as file:
    json.dump(files, file, indent=4)

for fn in files.values():
    assert f"J0_{J0}" in fn
    assert f"J1_{J1}" in fn

correlations = {}

# Load the MPS and calculate the correlations.
for method in files.keys():
    if method == "target":
        with open(files[method], "rb") as mps_f:
            raw_mps = pkl.load(mps_f)
        mps = qiskit_to_tenpy_mps(raw_mps)
        correlations["target"] = correlation_function(mps)
    elif method == "AQC":
        with open(files[method], "r") as file:
            qasm_string = file.read()

        qc = loads(qasm_string)

        qiskit_mps = tensornetwork_from_circuit(
            qc,
            AerSimulator(
                method="matrix_product_state",
                matrix_product_state_truncation_threshold=1e-16,
                matrix_product_state_max_bond_dimension=50,
            ),
        )
        mps = qiskit_to_tenpy_mps(qiskit_mps._as_tuple())
        correlations["AQC"] = correlation_function(mps)
    elif method == "Ran":
        for layers in [1, 10]:
            with open(files[method], mode="r") as file:
                metrics = json.load(file)

            qasm_string = metrics[str(layers)]["circuit"]
            qc = loads(qasm_string)

            qiskit_mps = tensornetwork_from_circuit(
                qc,
                AerSimulator(
                    method="matrix_product_state",
                    matrix_product_state_truncation_threshold=1e-16,
                    matrix_product_state_max_bond_dimension=50,
                ),
            )
            mps = qiskit_to_tenpy_mps(qiskit_mps._as_tuple())
            correlations[f"Ran_{layers}"] = correlation_function(mps)


# Plotting

styles = {
    "target": ["k", "x", 1.0],
    "AQC": ["tab:blue", "o", 0.8],
    "Ran_1": ["tab:green", "v", 0.8],
    "Ran_10": ["tab:red", "^", 0.8],
}

# Nearest-neighbour correlations
for key, value in correlations.items():
    plt.scatter(
        list(range(n - 1)),
        [value[i, i + 1] for i in range(n - 1)],
        color=styles[key][0],
        marker=styles[key][1],
        alpha=styles[key][2],
        label=key,
    )

plt.axhline(0, color="k")
plt.xlabel("Qubit, i")
plt.ylabel(r"$C_{zz}(i,i+1)$")
plt.title(f"Nearest-neigbour correlations: $J_0={J0}, J_1={J1}$")
plt.legend()
plt.savefig(
    os.path.join(plot_dir, f"nearest_neighbour_correlations.pdf"),
    dpi=500,
)
plt.clf()

# Specific-site correlations
for i in specific_indices:
    for key, value in correlations.items():

        min_index = max(i - 20, 0)
        max_index = min(i + 20, n - 1)
        plt.scatter(
            list(range(min_index, max_index + 1)),
            value[i, min_index : (max_index + 1)],
            color=styles[key][0],
            marker=styles[key][1],
            alpha=styles[key][2],
            label=key,
        )
    plt.axhline(0, color="k")
    plt.xlabel("Qubit, j")
    plt.ylabel(r"$C_{zz}(i,j)$")
    plt.title(f"Correlation function for site i={i}: $J_0={J0}, J_1={J1}$")
    plt.legend()
    plt.savefig(
        os.path.join(plot_dir, f"qubit_{i}_correlations.pdf"),
        dpi=500,
    )
    plt.clf()
