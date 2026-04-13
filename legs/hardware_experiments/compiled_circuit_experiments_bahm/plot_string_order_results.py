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
import matplotlib.pyplot as plt
from pathlib import Path
from adaptaqc.utils.utilityfunctions import qiskit_to_tenpy_mps
from legs.hardware_experiments.utils import plot_zne, get_observable_results
from legs.bond_alternating_heisenberg_model.utils import string_expectation_values
from legs.bond_alternating_heisenberg_model.utils import load_mps_from_file
from legs.hardware_experiments.compiled_circuit_experiments_bahm.constants import (
    EXPERIMENT_TO_PLOT,
)
from qiskit.quantum_info import SparsePauliOp

plt.rc("font", family="serif")
plt.rc("text", usetex=True)

red = "#FF644E"
blue = "#00A2FF"


n = 100
J0 = 1.0
J1 = -2.0

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

# Compiled circuit MPS
compiled_circuit_fn = experiments[str(EXPERIMENT_TO_PLOT)]["compiled_circuit_fn"]
assert f"J0_{J0}" in compiled_circuit_fn
assert f"J1_{J1}" in compiled_circuit_fn

circuit_mps = load_mps_from_file(compiled_circuit_fn)


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

# Save figures in the right directory
split_fn = result_fn.split("/")
results_index = split_fn.index("results")
plot_dir = "/".join(split_fn[:results_index]) + "/figures/string_order_plots"
Path(plot_dir).mkdir(parents=True, exist_ok=True)

# String order observables:
# Plot the string order over multiple windows by defining multiple values of s. NOTE: you can only
# use values of s which were included in the job submission script.
s_list = [20, 30, 40, 50, 60]
num_obs = 10
avoid = min(20, n - max(s_list) - 2 * num_obs - 1)

ms = 8
lw = 1.5
cap = 3

for s in s_list:
    tenpy_evs = string_expectation_values(tenpy_mps, s=s, avoid=avoid)
    circuit_evs = string_expectation_values(circuit_mps, s=min(s_list), avoid=avoid)

    even_string_order_observables = [
        SparsePauliOp.from_sparse_list(
            [("Z" * 2 * (j + 1), list(range(s, s + 2 * (j + 1))), (-1) ** (j + 1))], n
        )
        for j in range(num_obs)
    ]
    odd_string_order_observables = [
        SparsePauliOp.from_sparse_list(
            [
                (
                    "Z" * 2 * (j + 1),
                    list(range(s + 1, s + 2 * (j + 1) + 1)),
                    (-1) ** (j + 1),
                )
            ],
            n,
        )
        for j in range(num_obs)
    ]

    even_results = get_observable_results(even_string_order_observables, result_dict)
    odd_results = get_observable_results(odd_string_order_observables, result_dict)

    # fig_0: String order plots.
    plt.rcParams.update({"font.size": 12})
    fig_0, axs_0 = plt.subplots(1, 2)

    num_evs = even_results["evs"].shape[0]
    x = list(range(num_evs))

    # List of (axis, values, errors, label, colour, marker), for all combinations of:
    # (even, odd) X (DMRG, circuit MPS, hardware no ZNE, hardware ZNE).
    plot_list = [
        (0, tenpy_evs[0][:num_evs], None, "DMRG", "k", "x"),
        (0, circuit_evs[0][:num_evs], None, "Compiled circuit:\nMPS simulation", "tab:green", "+"),
        (
            0,
            even_results["evs_noise_factors"][:, 0],
            even_results["stds_noise_factors"][:, 0],
            f"{backend_name} no ZNE",
            red,
            "x",
        ),
        (
            0,
            even_results["evs"],
            even_results["stds"],
            f"{backend_name} ZNE",
            blue,
            "x",
        ),
        (1, tenpy_evs[1][:num_evs], None, "DMRG", "k", "x"),
        (1, circuit_evs[1][:num_evs], None, "Compiled circuit:\nMPS simulation", "tab:green", "+"),
        (
            1,
            odd_results["evs_noise_factors"][:, 0],
            odd_results["stds_noise_factors"][:, 0],
            f"{backend_name} no ZNE",
            red,
            "x",
        ),
        (1, odd_results["evs"], odd_results["stds"], f"{backend_name} ZNE", blue, "x"),
    ]

    lines = [
        (-1, "--"),
        (1, "--"),
        (0, "-"),
    ]
    for i, values, errors, label, color, marker in plot_list:
        axs_0[i].errorbar(
            x,
            values,
            errors,
            marker=marker,
            ms=ms,
            color=color,
            linestyle="",
            lw=lw,
            mew=lw,
            capsize=cap,
            label=label,
        )

    [axs_0[0].axhline(line[0], color="k", linestyle=line[1]) for line in lines]
    axs_0[0].set_xlabel("Index, i")
    axs_0[0].set_title("Even")
    [axs_0[1].axhline(line[0], color="k", linestyle=line[1]) for line in lines]
    axs_0[1].set_xlabel("Index, i")
    axs_0[1].set_title("Odd")

    axs_0[0].set_ylabel(r"$(-1)^{i+1} \langle Z_s ... Z_{s+2i+1} \rangle$")
    axs_0[1].legend(loc="lower right", prop={"size": 12})

    fig_0.suptitle(
        f"$J_0={J0}, J_1={J1}$ string order\n$s={s}(e),\,{s+1}(o)$",
        fontsize=16,
    )

    fig_0.tight_layout()
    fig_0.savefig(os.path.join(plot_dir, f"string_order_plots_s_{s}.pdf"), dpi=500)

    # fig_1: individual ZNE plots.
    fig_1, axs_1 = plot_zne(
        result_dict,
        even_results["indices"] + odd_results["indices"],
        tenpy_evs[0][:num_obs] + tenpy_evs[1][:num_obs],
        hlines=[-1, 1],
        scale=1,
    )

    fig_1.suptitle(f"$J_0={J0}, J_1={J1}$", fontsize=16)

    fig_1.tight_layout()
    fig_1.subplots_adjust(top=0.95)

    fig_1.savefig(
        os.path.join(plot_dir, f"individual_string_order_ZNE_plots_s_{s}.pdf"), dpi=500
    )
