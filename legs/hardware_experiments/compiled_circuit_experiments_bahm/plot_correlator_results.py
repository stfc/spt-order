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
from matplotlib.colors import SymLogNorm
from pathlib import Path
from adaptaqc.utils.utilityfunctions import qiskit_to_tenpy_mps
from legs.hardware_experiments.utils import (
    get_observable_results,
    calculate_correlators_and_errors,
)
from legs.hardware_experiments.compiled_circuit_experiments_bahm.constants import (
    EXPERIMENT_TO_PLOT,
)
from qiskit.quantum_info import SparsePauliOp

plt.rc("font", family="serif")
plt.rc("text", usetex=True)

n = 100
J0 = 1.0
J1 = -2.0

# Specify the sites i for which to plot Czz(i, j) as individual plots
specific_indices = [0, 50]


# File paths
# Add the result and target MPS file paths to filepaths.json.
with open("filepaths.json", mode="r") as file:
    experiments = json.load(file)

# Target MPS
target_mps_fn = experiments[str(EXPERIMENT_TO_PLOT)]["target_mps_fn"]
assert f"J0_{J0}" in target_mps_fn
assert f"J1_{J1}" in target_mps_fn
with open(target_mps_fn, "rb") as mps_f:
    raw_target_mps = pkl.load(mps_f)

tenpy_mps = qiskit_to_tenpy_mps(raw_target_mps)
tenpy_z = tenpy_mps.expectation_value("Sz")

# Correlators
correlators_tenpy = np.empty((n, n))
correlators_tenpy[:] = np.nan

for i in range(n):
    for j in range(i + 1, min(i + 20, n - 1) + 1):
        tenpy_zz = tenpy_mps.expectation_value_term([("Sz", i), ("Sz", j)])
        tenpy_correlator = tenpy_zz - tenpy_z[i] * tenpy_z[j]
        correlators_tenpy[i, j] = tenpy_correlator
        correlators_tenpy[j, i] = tenpy_correlator


# Hardware results
result_fn = experiments[str(EXPERIMENT_TO_PLOT)]["result_fn"]
assert f"J0_{J0}" in result_fn
assert f"J1_{J1}" in result_fn
with open(result_fn, "rb") as file:
    result_dict = pkl.load(file)

# Observables:
# Zi on all qubits.
z_observables = [SparsePauliOp.from_sparse_list([("Z", [i], 1)], n) for i in range(n)]

# ZiZj with i=0,...,99 and j=i+1,...,i+20 (obviously not going beyond 99).
zz_observables = []
for i in range(n):
    max_index = min(i + 20, n - 1)
    for j in range(i + 1, max_index + 1):
        zz_observables.append(SparsePauliOp.from_sparse_list([("ZZ", [i, j], 1)], n))

z_results = get_observable_results(z_observables, result_dict)
zz_results = get_observable_results(zz_observables, result_dict)

# Correlators: <S^z_i S^z_j> - <S^z_i> * <S^z_j>
# Extrapolated.
correlators, correlators_err = calculate_correlators_and_errors(
    z_results["evs"],
    zz_results["evs"],
    z_results["stds"],
    zz_results["stds"],
    zz_observables,
)
# Raw values (noise-factor-1)
raw_correlators, raw_correlators_err = calculate_correlators_and_errors(
    z_results["evs_noise_factors"][:, 0],
    zz_results["evs_noise_factors"][:, 0],
    z_results["stds_noise_factors"][:, 0],
    zz_results["stds_noise_factors"][:, 0],
    zz_observables,
)

# Save figures in the right directory
split_fn = result_fn.split("/")
results_index = split_fn.index("results")
plot_dir = "/".join(split_fn[:results_index]) + "/figures/correlation_plots"
Path(plot_dir).mkdir(parents=True, exist_ok=True)


minimum = min(np.nanmin(correlators), np.nanmin(correlators_tenpy))
maximum = max(np.nanmax(correlators), np.nanmax(correlators_tenpy))
if abs(minimum) > abs(maximum):
    vmin = minimum
    vmax = -minimum
else:
    vmin = -maximum
    vmax = maximum

norm = SymLogNorm(linthresh=1e-3, vmin=vmin, vmax=vmax)

# Hardware correlation heatmap
# Mitigated (ZNE).
extent = [0.5, n + 0.5, n + 0.5, 0.5]
fig = plt.imshow(correlators, cmap="RdBu", norm=norm, extent=extent)

plt.colorbar(fig, orientation="vertical", label=r"$C_{zz}(i,j)$")
plt.xlabel(r"Site index: $i$")
plt.ylabel(r"Site index: $j$")
plt.title(f"Mitigated hardware correlations: $J_0={J0}, J_1={J1}$")
plt.savefig(os.path.join(plot_dir, "mitigated_hardware_correlations.pdf"), dpi=500)
plt.clf()

# Unmitigated (noise-factor-1) results.
fig = plt.imshow(raw_correlators, cmap="RdBu", norm=norm, extent=extent)

plt.colorbar(fig, orientation="vertical", label=r"$C_{zz}(i,j)$")
plt.xlabel(r"Site index: $i$")
plt.ylabel(r"Site index: $j$")
plt.title(f"Unmitigated hardware correlations: $J_0={J0}, J_1={J1}$")
plt.savefig(os.path.join(plot_dir, "unmitigated_hardware_correlations.pdf"), dpi=500)
plt.clf()

# Tenpy correlation heatmap
fig = plt.imshow(correlators_tenpy, cmap="RdBu", norm=norm, extent=extent)

plt.colorbar(fig, orientation="vertical", label=r"$C_{zz}(i,j)$")
plt.xlabel(r"Site index: $i$")
plt.ylabel(r"Site index: $j$")
plt.title(f"TeNPy correlations: $J_0={J0}, J_1={J1}$")
plt.savefig(os.path.join(plot_dir, "tenpy_correlations.pdf"), dpi=500)
plt.clf()

# Nearest-neighbour correlations
# Mitigated (ZNE).
plt.errorbar(
    list(range(n - 1)),
    [correlators[i, i + 1] for i in range(n - 1)],
    [correlators_err[i, i + 1] for i in range(n - 1)],
    color="tab:blue",
    marker="x",
    linestyle="",
    label="Mitigated hardware",
)
# Unmitigated (noise-factor-1) results.
plt.errorbar(
    list(range(n - 1)),
    [raw_correlators[i, i + 1] for i in range(n - 1)],
    [raw_correlators_err[i, i + 1] for i in range(n - 1)],
    color="tab:orange",
    marker="x",
    linestyle="",
    label="Unmitigated hardware",
)
# TeNPy.
plt.scatter(
    list(range(n - 1)),
    [correlators_tenpy[i, i + 1] for i in range(n - 1)],
    color="k",
    marker="x",
    label="TeNPy",
)
plt.axhline(0, color="k")
plt.xlabel("Qubit, i")
plt.ylabel(r"$C_{zz}(i,i+1)$")
plt.title(f"Nearest-neigbour correlations: $J_0={J0}, J_1={J1}$")
plt.legend()
plt.savefig(os.path.join(plot_dir, "nearest_neighbour_correlations.pdf"), dpi=500)
plt.clf()

# Specific-site correlations
for i in specific_indices:

    min_index = max(i - 20, 0)
    max_index = min(i + 20, n - 1)
    x = list(range(min_index, max_index + 1))
    # Mitigated (ZNE).
    plt.errorbar(
        list(range(min_index, max_index + 1)),
        correlators[i, min_index : (max_index + 1)],
        correlators_err[i, min_index : (max_index + 1)],
        color="tab:blue",
        marker="x",
        linestyle="",
        label="Mitigated hardware",
    )
    # Unmitigated (noise-factor-1) results.
    plt.errorbar(
        list(range(min_index, max_index + 1)),
        raw_correlators[i, min_index : (max_index + 1)],
        raw_correlators_err[i, min_index : (max_index + 1)],
        color="tab:orange",
        marker="x",
        linestyle="",
        label="Unmitigated hardware",
    )
    # TeNPy.
    plt.scatter(
        list(range(min_index, max_index + 1)),
        correlators_tenpy[i, min_index : (max_index + 1)],
        color="k",
        marker="x",
        label="TeNPy",
    )
    plt.axhline(0, color="k")
    plt.xlabel("Qubit, j")
    plt.ylabel(r"$C_{zz}(i,j)$")
    plt.title(f"Correlation function for site i={i}: $J_0={J0}, J_1={J1}$")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f"qubit_{i}_correlations.pdf"), dpi=500)
    plt.clf()
