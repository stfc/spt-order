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
import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from legs.hardware_experiments.utils import get_observable_results
from legs.bond_alternating_heisenberg_model.utils import load_mps_from_file
from legs.hardware_experiments.compiled_circuit_experiments_bahm.entanglement_spectrum.utils import (
    find_eigenvalues,
)
from qiskit.quantum_info import SparsePauliOp
from tenpy import Array

plt.rc("font", family="serif")
plt.rc("text", usetex=True)
plt.rcParams.update({"font.size": 12})

red = "#FF644E"
blue = "#00A2FF"

n = 100
J0 = 1.0
J1 = -1.0
experiment_to_plot = 1
bootstrap_samples = 1000
max_eigenvalues = 8
# Segment length.
l_list = [1, 2, 3, 4, 5, 6]
# The eigenvalue plot will always consist of at least two panels: unmitigated results (left), and
# noiseless MPS results (right). Add any combination of "best", "exponential", "polynomial_degree_2"
# and "linear" to add these results to middle panels.
extrapolator_list = []

# File paths
# Add the result and target MPS file paths to filepaths.json.
with open("tomography_filepaths.json", mode="r") as file:
    experiments = json.load(file)

# Target MPS
target_mps_fn = experiments[str(experiment_to_plot)]["target_mps_fn"]
assert f"J0_{J0}" in target_mps_fn
assert f"J1_{J1}" in target_mps_fn
tenpy_mps = load_mps_from_file(target_mps_fn)
print(max(tenpy_mps.chi))

# # Compiled circuit
# compiled_circuit_fn = experiments[str(experiment_to_plot)]["compiled_circuit_fn"]
# assert f"J0_{J0}" in compiled_circuit_fn
# assert f"J1_{J1}" in compiled_circuit_fn
# circuit_mps = load_mps_from_file(compiled_circuit_fn)


# Hardware results
result_fn = experiments[str(experiment_to_plot)]["result_fn"]
assert f"J0_{J0}" in result_fn
assert f"J1_{J1}" in result_fn
with open(result_fn, "rb") as file:
    result_dict = pkl.load(file)

# Get backend name from result_fn
split_fn = result_fn.split("_")
ibm_index = split_fn.index("ibm")
backend_name = split_fn[ibm_index] + "_" + split_fn[ibm_index + 1]

# True (False) if ZNE was (wasn't) used.
zne = result_dict["PrimitiveResult.metadata"]["resilience"]["zne_mitigation"]
if not zne:
    assert len(extrapolator_list) == 0

# Save figures in the right directory
split_fn = result_fn.split("/")
results_index = split_fn.index("results")
plot_dir = "/".join(split_fn[:results_index]) + "/figures/entanglement_spectrum_plots"
Path(plot_dir).mkdir(parents=True, exist_ok=True)

# Observables: all Pauli strings.
pauli_ops = ["I", "X", "Y", "Z"]

# Tenpy Pauli Z in SpinSite basis convention.
op_dict = {
    "Z": Array.from_ndarray_trivial(
        np.array([[-1, 0], [0, 1]]),
        labels=["p", "p*"],
    ),
    "Y": Array.from_ndarray_trivial(
        np.array([[0, 1j], [-1j, 0]]),
        labels=["p", "p*"],
    ),
    "X": Array.from_ndarray_trivial(
        np.array([[0, 1], [1, 0]]),
        labels=["p", "p*"],
    ),
    "I": Array.from_ndarray_trivial(
        np.array([[1, 0], [0, 1]]),
        labels=["p", "p*"],
    ),
}

# TODO: Remove hard-coded numbers
# fig, axs = plt.subplots(1, 4, sharey=True, figsize=(10, 2.5), layout="constrained")
fig, axs = plt.subplots(2, 3, sharey=True, layout="constrained")

for l in l_list:
    # TODO: Remove hard-coded numbers
    row = (l - 1) % 2
    col = (l - 1) // 2

    observables = [0] * 4**l
    observable_labels = [0] * 4**l
    tenpy_exp_vals = [0] * 4**l
    # noiseless_circuit_exp_vals = [0] * 4**l
    for k in range(4**l):
        if k == 0:
            indices = "0" * l
        else:
            # Count in base-4.
            pad = int(l - 1 - np.floor(np.log(k) / np.log(4)))
            indices = str(np.base_repr(k, 4, pad))[::-1]

        # Pauli string, big-endian.
        pauli_string = "".join([pauli_ops[int(i)] for i in indices])

        # Qiskit observable
        observables[k] = SparsePauliOp.from_sparse_list(
            [(pauli_string, range(l), 1)], n
        )
        observable_labels[k] = pauli_string

        # Tenpy expectation value
        tenpy_exp_vals[k] = tenpy_mps.expectation_value_multi_sites(
            operators=[op_dict[c] for c in pauli_string], i0=0
        )
        # # Circuit MPS expectation value
        # noiseless_circuit_exp_vals[k] = circuit_mps.expectation_value_multi_sites(
        #     operators=[op_dict[c] for c in pauli_string], i0=0
        # )

    # Extract hardware results.
    hardware_results = get_observable_results(observables, result_dict)

    # Find eigenvalues (eig_vals, eig_vals_mean, eig_vals_stds) for all methods.
    # TeNPy
    tenpy_eigenvalues = find_eigenvalues(l, observables, tenpy_exp_vals)

    # # Noiseless circuit
    # noiseless_circuit_eigenvalues = find_eigenvalues(
    #     l, observables, noiseless_circuit_exp_vals
    # )

    # Unmitigated hardware (no ZNE).
    raw_eigenvalues = find_eigenvalues(
        l,
        observables,
        hardware_results["evs_noise_factors"][:, 0] if zne else hardware_results["evs"],
        (
            hardware_results["stds_noise_factors"][:, 0]
            if zne
            else hardware_results["stds"]
        ),
        bootstrap_samples,
    )

    # # Plot entanglement spectrum.
    # size = figaspect(3 / (len(extrapolator_list) + 3))
    # fig_0, axs_0 = plt.subplots(1, len(extrapolator_list) + 2, sharey=True, layout="constrained", figsize=size)

    # Left plot: unmitigated data.
    # NOTE: height: Qiskit EV, errorbar: bootstrapping
    print(f"Max diff: {np.max(np.abs(raw_eigenvalues[0] - raw_eigenvalues[1]))}")

    x = list(range(1, 2**l + 1))[:max_eigenvalues]
    axs[row, col].bar(
        x,
        height=raw_eigenvalues[0][:max_eigenvalues],
        yerr=raw_eigenvalues[2][:max_eigenvalues],
        capsize=3,
        color=red if row == 0 else blue,
        label=f"{backend_name}, {'odd $l$' if row == 0 else 'even $l$'}",
    )
    # axs[row, col].scatter(list(range(2**l))[:max_eigenvalues], raw_eigenvalues[0][:max_eigenvalues], marker="x", label="No bootstrap")

    axs[row, col].scatter(
        x,
        tenpy_eigenvalues[0][:max_eigenvalues],
        color="k",
        marker="x",
        label="DMRG",
    )

    x_labels = [f"$\lambda_{'{' + str(i) + '}'}$" for i in x]
    axs[row, col].set_xticks(x, x_labels)

    axs[row, col].set_title(f"$l=$ {l}")

# fig.suptitle(f"Reduced density matrix eigenvalues for $l$ sites. {bootstrap_samples} bootstrap samples.")
# fig.supxlabel(r"$i^{th}$ largest eigenvalue")
# fig.supylabel(r"RDM ($\rho^{(l)}$) eigenvalues")

axs[row, col].set_ylim(bottom=0)

# Empty bar so the legend shows orange and blue bars.
axs[row, col].bar(
    [1],
    [0],
    color=blue if row == 0 else red,
    label=f"{backend_name}, {'even $l$' if row == 0 else 'odd $l$'}",
)
axs[row, col].legend(fontsize=8)

fn = os.path.join(
    plot_dir,
    f"J0_{J0}_J1_{J1}_composite_l_{l}_entanglement_spectrum_{bootstrap_samples}_samples_{"_".join(extrapolator_list)}.pdf",
)
# fig.tight_layout()
fig.savefig(fn, dpi=500)
print(f"Figure saved to: {fn}")
