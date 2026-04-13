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
from matplotlib.figure import figaspect
from pathlib import Path
from legs.hardware_experiments.utils import get_observable_results
from legs.hardware_experiments.compiled_circuit_experiments_bahm.entanglement_spectrum.utils import (
    find_eigenvalues,
)
from legs.bond_alternating_heisenberg_model.utils import load_mps_from_file
from qiskit.quantum_info import SparsePauliOp
from tenpy import Array

plt.rc("font", family="serif")
plt.rc("text", usetex=True)
plt.rcParams.update({"font.size": 8})


n = 100
J0 = 1.0
J1 = -1.0
experiment_to_plot = 1
bootstrap_samples = 1000
# Segment length.
l_list = [1, 2, 3, 4]
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

# Compiled circuit
compiled_circuit_fn = experiments[str(experiment_to_plot)]["compiled_circuit_fn"]
assert f"J0_{J0}" in compiled_circuit_fn
assert f"J1_{J1}" in compiled_circuit_fn
circuit_mps = load_mps_from_file(compiled_circuit_fn)


# Hardware results
result_fn = experiments[str(experiment_to_plot)]["result_fn"]
assert f"J0_{J0}" in result_fn
assert f"J1_{J1}" in result_fn
with open(result_fn, "rb") as file:
    result_dict = pkl.load(file)

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

for l in l_list:
    observables = [0] * 4**l
    observable_labels = [0] * 4**l
    tenpy_exp_vals = [0] * 4**l
    noiseless_circuit_exp_vals = [0] * 4**l
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
        # Circuit MPS expectation value
        noiseless_circuit_exp_vals[k] = circuit_mps.expectation_value_multi_sites(
            operators=[op_dict[c] for c in pauli_string], i0=0
        )

    # Extract hardware results.
    hardware_results = get_observable_results(observables, result_dict)

    # Find eigenvalues (eig_vals, eig_vals_mean, eig_vals_stds) for all methods.
    # TeNPy
    tenpy_eigenvalues = find_eigenvalues(l, observables, tenpy_exp_vals)

    # Noiseless circuit
    noiseless_circuit_eigenvalues = find_eigenvalues(
        l, observables, noiseless_circuit_exp_vals
    )

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

    # Plot entanglement spectrum.
    size = figaspect(3 / (len(extrapolator_list) + 3))
    fig_0, axs_0 = plt.subplots(
        1, len(extrapolator_list) + 2, sharey=True, layout="constrained", figsize=size
    )

    # Left plot: unmitigated data.
    axs_0[0].bar(
        range(2**l),
        height=raw_eigenvalues[1],
        yerr=raw_eigenvalues[2],
        label=f"Bootstrap\n{bootstrap_samples} samples",
    )
    axs_0[0].scatter(range(2**l), raw_eigenvalues[0], marker="x", label="No bootstrap")
    axs_0[0].set_title("Unmitigated")
    axs_0[0].legend(fontsize=6)
    axs_0[0].set_ylabel(r"RDM ($\rho^{(l)}$) eigenvalues")

    # Right plot: noiseless MPS data.
    final_index = len(extrapolator_list) + 1
    axs_0[final_index].bar(range(2**l), height=tenpy_eigenvalues[0], label="Target MPS")
    axs_0[final_index].scatter(
        range(2**l),
        noiseless_circuit_eigenvalues[0],
        color="k",
        marker="x",
        label="Compiled circuit",
    )
    axs_0[final_index].set_title("Noiseless\n(MPS)")
    axs_0[final_index].legend(fontsize=6)

    # Middle plots: various extrapolations.
    for i, extrapolator in enumerate(extrapolator_list):
        # Mitigated hardware (ZNE).
        if extrapolator == "best":
            extrapolated_exp_vals = hardware_results["evs"]
            extrapolated_stds = hardware_results["stds"]
        else:
            index = result_dict["PrimitiveResult.metadata"]["resilience"]["zne"][
                "extrapolator"
            ].index(extrapolator)
            extrapolated_exp_vals = hardware_results["evs_extrapolated"][:, index, 0]
            extrapolated_stds = hardware_results["stds_extrapolated"][:, index, 0]

        extrapolated_eigenvalues = find_eigenvalues(
            l, observables, extrapolated_exp_vals, extrapolated_stds, bootstrap_samples
        )

        axs_0[i + 1].bar(
            range(2**l),
            height=extrapolated_eigenvalues[1],
            yerr=extrapolated_eigenvalues[2],
        )
        axs_0[i + 1].scatter(range(2**l), extrapolated_eigenvalues[0], marker="x")
        axs_0[i + 1].set_title(f"Mitigated\n{extrapolator}")

        # Plot individual expectation values.
        fig_1, axs_1 = plt.subplots(1, 1)
        # Extrapolated
        axs_1.errorbar(
            list(range(4**l)),
            extrapolated_exp_vals,
            extrapolated_stds,
            label=f"Extrapolator:\n{'Qiskit best fit' if extrapolator is None else extrapolator}",
            linestyle="",
            marker="x",
        )
        # Raw values
        axs_1.errorbar(
            list(range(4**l)),
            hardware_results["evs_noise_factors"][:, 0],
            hardware_results["stds_noise_factors"][:, 0],
            label="Raw values",
            linestyle="",
            marker="x",
        )
        # TeNPy values
        axs_1.scatter(
            list(range(4**l)),
            tenpy_exp_vals,
            color="k",
            marker="x",
            label="TeNPy target MPS",
        )
        axs_1.axhline(0, color="k", linestyle="--")
        if l < 3:
            rotation = 0
            size = "small"
            interval = 1
        else:
            rotation = 90
            size = "xx-small"
            # Skip some tick labels so they don't overlap.
            interval = int(np.ceil(4**l / 4**3))
        axs_1.set_xticks(
            list(range(4**l))[::interval],
            observable_labels[::interval],
            rotation=rotation,
            size=size,
        )
        axs_1.set_xlabel(
            f"Pauli string (big-endian) ({'I' * l}, {'X' + 'I' * (l-1)}, ..., {'Z' * l})"
        )
        axs_1.set_ylabel("Expectation value")
        axs_1.set_title(f"$l=${l}-qubit Pauli string expectation values.")
        axs_1.legend()

        fig_1.tight_layout()
        fn = os.path.join(
            plot_dir,
            f"l_{l}_individual_expectation_values_{extrapolator}.pdf",
        )
        fig_1.savefig(fn, dpi=500)
        print(f"Figure saved to: {fn}")

    if not zne:
        # Plot individual EVs even if no extrapolator is used.
        fig_1, axs_1 = plt.subplots(1, 1)

        # Raw values
        axs_1.errorbar(
            list(range(4**l)),
            hardware_results["evs"],
            hardware_results["stds"],
            label="Raw values",
            color="tab:orange",
            linestyle="",
            marker="x",
        )
        # TeNPy values
        axs_1.scatter(
            list(range(4**l)),
            tenpy_exp_vals,
            color="k",
            marker="x",
            label="TeNPy target MPS",
        )
        axs_1.axhline(0, color="k", linestyle="--")
        if l < 3:
            rotation = 0
            size = "small"
            interval = 1
        else:
            rotation = 90
            size = "xx-small"
            # Skip some tick labels so they don't overlap.
            interval = int(np.ceil(4**l / 4**3))
        axs_1.set_xticks(
            list(range(4**l))[::interval],
            observable_labels[::interval],
            rotation=rotation,
            size=size,
        )
        axs_1.set_xlabel(
            f"Pauli string (big-endian) ({'I' * l}, {'X' + 'I' * (l-1)}, ..., {'Z' * l})"
        )
        axs_1.set_ylabel("Expectation value")
        axs_1.set_title(f"$l=${l}-qubit Pauli string expectation values.")
        axs_1.legend()

        fig_1.tight_layout()
        fn = os.path.join(
            plot_dir,
            f"l_{l}_individual_expectation_values_no_ZNE.pdf",
        )
        fig_1.savefig(fn, dpi=500)
        print(f"Figure saved to: {fn}")

    [
        axs_0[i].axhline(0, color="k", linestyle="--")
        for i in range(len(extrapolator_list) + 2)
    ]
    fig_0.suptitle(f"Reduced density matrix eigenvalues for $l=${l} sites.")
    fig_0.supxlabel(r"$i^{th}$ largest eigenvalue")

    fn = os.path.join(
        plot_dir,
        f"l_{l}_entanglement_spectrum_{bootstrap_samples}_samples_{"_".join(extrapolator_list)}.pdf",
    )
    fig_0.savefig(fn, dpi=500)
    print(f"Figure saved to: {fn}")
