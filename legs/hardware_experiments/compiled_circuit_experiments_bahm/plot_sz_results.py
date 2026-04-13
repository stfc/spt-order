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
from legs.hardware_experiments.utils import plot_zne, get_observable_results
from legs.bond_alternating_heisenberg_model.utils import load_mps_from_file
from legs.hardware_experiments.compiled_circuit_experiments_bahm.constants import (
    EXPERIMENT_TO_PLOT,
)
from qiskit.quantum_info import SparsePauliOp

plt.rc("font", family="serif")
plt.rc("text", usetex=True)
plt.rcParams.update({"font.size": 12})

red = "#FF644E"
blue = "#00A2FF"


n = 100
J0 = 1.0
J1 = -2.0

# If `num_sites` is not None, two separate panels will be plotted, showing <Sz> for `num_sites` from
# each end of the chain
num_sites = 20

# Set to None to use Qiskit "best" extrapolation, else choose extrapolator.
extrapolator = "linear"

# File paths
# Add the result and target MPS file paths to filepaths.json.
with open("filepaths.json", mode="r") as file:
    experiments = json.load(file)

# Target MPS
target_mps_fn = experiments[str(EXPERIMENT_TO_PLOT)]["target_mps_fn"]
assert f"J0_{J0}" in target_mps_fn
assert f"J1_{J1}" in target_mps_fn

tenpy_mps = load_mps_from_file(target_mps_fn)
tenpy_sz = tenpy_mps.expectation_value("Sz")

# Compiled circuit
compiled_circuit_fn = experiments[str(EXPERIMENT_TO_PLOT)]["compiled_circuit_fn"]
assert f"J0_{J0}" in compiled_circuit_fn
assert f"J1_{J1}" in compiled_circuit_fn

circuit_mps = load_mps_from_file(compiled_circuit_fn)
circuit_sz = circuit_mps.expectation_value("Sz")

# Magnetisation
tenpy_mag = np.sum(tenpy_sz) / n

print(f"Tenpy magnetisation:    {np.round(tenpy_mag, 3)}")


# Hardware results
result_fn = experiments[str(EXPERIMENT_TO_PLOT)]["result_fn"]
assert f"J0_{J0}" in result_fn
assert f"J1_{J1}" in result_fn
with open(result_fn, "rb") as file:
    result_dict = pkl.load(file)

# Get backend name from result_fn
split_fn = result_fn.split("_")
ibm_index = split_fn.index("ibm")
backend_name = split_fn[ibm_index] + "_" + split_fn[ibm_index + 1]

# Observables:
# Zi on all qubits.
z_observables = [SparsePauliOp.from_sparse_list([("Z", [i], 1)], n) for i in range(n)]
z_results = get_observable_results(z_observables, result_dict)

# Get extrapolated EVs
if extrapolator is None:
    extrapolated_evs = z_results["evs"] / 2
    extrapolated_stds = z_results["stds"] / 2
else:
    extrapolator_index = result_dict["PrimitiveResult.metadata"]["resilience"]["zne"][
        "extrapolator"
    ].index(extrapolator)
    extrapolated_evs = z_results["evs_extrapolated"][:, extrapolator_index, 0] / 2
    extrapolated_stds = z_results["stds_extrapolated"][:, extrapolator_index, 0] / 2

# Magnetisation: sum(<S^z_i>)
# Error calculated using: e_{a+b} = sqrt((e_a)^2 + (e_b)^2)
mag = np.sum(extrapolated_evs) / n
mag_err = np.sqrt(np.sum(np.square(extrapolated_stds))) / n

print(f"Hardware magnetisation: {np.round(mag, 3)} +/- {np.round(mag_err, 3)}")

# Save figures in the right directory
split_fn = result_fn.split("/")
results_index = split_fn.index("results")
plot_dir = "/".join(split_fn[:results_index]) + "/figures/sz_plots"
Path(plot_dir).mkdir(parents=True, exist_ok=True)

lines = [
    # (-0.5, "--"),
    # (0.5, "--"),
    (0, "-"),
]

ms = 8
lw = 1.5
cap = 2

# Sz plot:
if num_sites is None:
    fig_0, axs_0 = plt.subplots(1)
    # TeNPy.
    axs_0.scatter(
        list(range(n)), tenpy_sz, marker="x", s=ms**2, color="k", lw=lw, label="DMRG"
    )
    # Circuit MPS.
    axs_0.scatter(
        list(range(n)),
        circuit_sz,
        marker="+",
        lw=lw,
        s=ms**2,
        color="tab:green",
        label="Compiled circuit:\nMPS simulation",
    )
    # Unmitigated (noise-factor-1) results.
    axs_0.errorbar(
        list(range(n)),
        z_results["evs_noise_factors"][:, 0] / 2,
        z_results["stds_noise_factors"][:, 0] / 2,
        marker="x",
        ms=ms,
        color=red,
        linestyle="",
        lw=lw,
        mew=lw,
        capsize=cap,
        label=f"{backend_name} no ZNE",
    )
    # Mitigated results (ZNE).
    axs_0.errorbar(
        list(range(n)),
        extrapolated_evs,
        extrapolated_stds,
        marker="x",
        ms=ms,
        color=blue,
        linestyle="",
        lw=lw,
        mew=lw,
        capsize=cap,
        label=f"{backend_name} ZNE",
    )

    [axs_0.axhline(line[0], color="k", linestyle=line[1]) for line in lines]
    axs_0.set_xlabel("Qubit, i")
    axs_0.set_ylabel(r"$\langle Z_i \rangle / 2$")
    axs_0.legend(loc="lower right", prop={"size": 12})
    fn = os.path.join(plot_dir, f"sz_plots_{extrapolator}_extrapolator.pdf")
else:
    fig_0, axs_0 = plt.subplots(1, 2, sharey=True)
    # Left `num_sites`
    # TeNPy.
    axs_0[0].scatter(
        list(range(num_sites)),
        tenpy_sz[:num_sites],
        marker="x",
        s=ms**2,
        lw=lw,
        color="k",
        label="DMRG",
    )
    # Circuit MPS.
    axs_0[0].scatter(
        list(range(num_sites)),
        circuit_sz[:num_sites],
        marker="+",
        s=ms**2,
        lw=lw,
        color="tab:green",
        label="Compiled circuit:\nMPS simulation",
    )
    # Unmitigated (noise-factor-1) results.
    axs_0[0].errorbar(
        list(range(num_sites)),
        z_results["evs_noise_factors"][:, 0][:num_sites] / 2,
        z_results["stds_noise_factors"][:, 0][:num_sites] / 2,
        marker="x",
        ms=ms,
        color=red,
        linestyle="",
        lw=lw,
        mew=lw,
        capsize=cap,
        label=f"{backend_name} no ZNE",
    )
    # Mitigated results (ZNE).
    axs_0[0].errorbar(
        list(range(num_sites)),
        extrapolated_evs[:num_sites],
        extrapolated_stds[:num_sites],
        marker="x",
        ms=ms,
        color=blue,
        linestyle="",
        lw=lw,
        mew=lw,
        capsize=cap,
        label=f"{backend_name} ZNE",
    )
    [axs_0[0].axhline(line[0], color="k", linestyle=line[1]) for line in lines]
    axs_0[0].set_xlabel("Qubit, i")
    axs_0[0].set_ylabel(r"$\langle Z_i \rangle / 2$")

    # Left `num_sites`
    # TeNPy.
    axs_0[1].scatter(
        list(range(n - num_sites, n)),
        tenpy_sz[-num_sites:],
        marker="x",
        s=ms**2,
        lw=lw,
        color="k",
        label="DMRG",
    )
    # Circuit MPS.
    axs_0[1].scatter(
        list(range(n - num_sites, n)),
        circuit_sz[-num_sites:],
        marker="+",
        s=ms**2,
        lw=lw,
        color="tab:green",
        label="Compiled circuit:\nMPS simulation",
    )
    # Unmitigated (noise-factor-1) results.
    axs_0[1].errorbar(
        list(range(n - num_sites, n)),
        z_results["evs_noise_factors"][:, 0][-num_sites:] / 2,
        z_results["stds_noise_factors"][:, 0][-num_sites:] / 2,
        marker="x",
        ms=ms,
        color=red,
        linestyle="",
        lw=lw,
        mew=lw,
        capsize=cap,
        label=f"{backend_name} no ZNE",
    )
    # Mitigated results (ZNE).
    axs_0[1].errorbar(
        list(range(n - num_sites, n)),
        extrapolated_evs[-num_sites:],
        extrapolated_stds[-num_sites:],
        marker="x",
        ms=ms,
        color=blue,
        linestyle="",
        lw=lw,
        mew=lw,
        capsize=cap,
        label=f"{backend_name} ZNE",
    )

    axs_0[1].legend(loc="upper left")  # , prop={"size": 8})

    [axs_0[1].axhline(line[0], color="k", linestyle=line[1]) for line in lines]
    axs_0[1].set_xlabel("Qubit, i")
    fn = os.path.join(
        plot_dir,
        f"sz_plots_{num_sites}_sites_from_edge_{extrapolator}_extrapolator.pdf",
    )

# fig_0.suptitle(
#     f"$J_0={J0}, J_1={J1}$\nTeNPy: $M={np.round(tenpy_mag, 3)}$"
#     + f"\n Hardware: $M={np.round(mag, 3)}\pm{np.round(mag_err, 3)}$,",
#     fontsize=16,
# )
fig_0.suptitle(f"$J_0={J0}, J_1={J1}$")

fig_0.tight_layout()
fig_0.savefig(fn, dpi=500)
print(f"Saved to: {fn}")

# fig_1: individual ZNE plots.
fig_1, axs_1 = plot_zne(
    result_dict, z_results["indices"], tenpy_sz, hlines=[-0.5, 0.5], scale=0.5
)

fig_1.suptitle(f"$J_0={J0}, J_1={J1}$", fontsize=16)

fig_1.tight_layout()
fig_1.subplots_adjust(top=0.95)

fn = os.path.join(plot_dir, "individual_ZNE_plots.pdf")
fig_1.savefig(fn, dpi=500)
print(f"Saved to: {fn}")
