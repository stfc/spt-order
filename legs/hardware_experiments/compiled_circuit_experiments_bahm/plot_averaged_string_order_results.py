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
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from legs.hardware_experiments.utils import get_observable_results
from legs.bond_alternating_heisenberg_model.utils import string_expectation_values
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

# File paths
# Add the result and target MPS file paths to filepaths.json.
with open("filepaths.json", mode="r") as file:
    experiments = json.load(file)

# Target MPS
target_mps_fn = experiments[str(EXPERIMENT_TO_PLOT)]["target_mps_fn"]
assert f"J0_{J0}" in target_mps_fn
assert f"J1_{J1}" in target_mps_fn

tenpy_mps = load_mps_from_file(target_mps_fn)

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

string_order = {
    "even": {
        "raw": {"evs": np.zeros(num_obs), "stds": np.zeros(num_obs)},
        "mitigated": {"evs": np.zeros(num_obs), "stds": np.zeros(num_obs)},
    },
    "odd": {
        "raw": {"evs": np.zeros(num_obs), "stds": np.zeros(num_obs)},
        "mitigated": {"evs": np.zeros(num_obs), "stds": np.zeros(num_obs)},
    },
}

tenpy_evs = string_expectation_values(tenpy_mps, s=min(s_list), avoid=avoid)
circuit_evs = string_expectation_values(circuit_mps, s=min(s_list), avoid=avoid)

for s in s_list:
    # Define the SparsePauliOps for all the observables.
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

    # Extract the hardware results for the observables
    even_results = get_observable_results(even_string_order_observables, result_dict)
    odd_results = get_observable_results(odd_string_order_observables, result_dict)

    # Average over each value of s.
    string_order["even"]["raw"]["evs"] += even_results["evs_noise_factors"][:, 0] / len(
        s_list
    )
    string_order["even"]["mitigated"]["evs"] += even_results["evs"] / len(s_list)
    string_order["odd"]["raw"]["evs"] += odd_results["evs_noise_factors"][:, 0] / len(
        s_list
    )
    string_order["odd"]["mitigated"]["evs"] += odd_results["evs"] / len(s_list)

    # Propagate the error using the error on a sum: std(a + b) = sqrt(std(a)**2 + std(b)**2).
    string_order["even"]["raw"]["stds"] += (
        even_results["stds_noise_factors"][:, 0] / len(s_list)
    ) ** 2
    string_order["even"]["mitigated"]["stds"] += (
        even_results["stds"] / len(s_list)
    ) ** 2
    string_order["odd"]["raw"]["stds"] += (
        odd_results["stds_noise_factors"][:, 0] / len(s_list)
    ) ** 2
    string_order["odd"]["mitigated"]["stds"] += (odd_results["stds"] / len(s_list)) ** 2

# Take the square root of the stds
string_order["even"]["raw"]["stds"] = np.sqrt(string_order["even"]["raw"]["stds"])
string_order["even"]["mitigated"]["stds"] = np.sqrt(
    string_order["even"]["mitigated"]["stds"]
)
string_order["odd"]["raw"]["stds"] = np.sqrt(string_order["odd"]["raw"]["stds"])
string_order["odd"]["mitigated"]["stds"] = np.sqrt(
    string_order["odd"]["mitigated"]["stds"]
)


# Averaged string order plot.
fig, axs = plt.subplots(1, 2, sharey=True)

x = [2 * (i + 1) for i in range(num_obs)]
ms = 8
lw = 1.5
cap = 3

# Odd
axs[0].errorbar(
    x,
    tenpy_evs[1][:num_obs],
    None,
    marker="x",
    ms=ms,
    color="k",
    linestyle="",
    lw=lw,
    mew=lw,
    capsize=cap,
    label="DMRG",
)
axs[0].errorbar(
    x,
    circuit_evs[1][:num_obs],
    None,
    marker="+",
    ms=ms,
    color="tab:green",
    linestyle="",
    lw=lw,
    mew=lw,
    capsize=cap,
    label="Compiled circuit:\nMPS simulation",
)
axs[0].errorbar(
    x,
    string_order["odd"]["raw"]["evs"],
    string_order["odd"]["raw"]["stds"],
    marker="x",
    ms=ms,
    color=red,
    linestyle="",
    lw=lw,
    mew=lw,
    capsize=cap,
    label=f"{backend_name} no ZNE",
)
axs[0].errorbar(
    x,
    string_order["odd"]["mitigated"]["evs"],
    string_order["odd"]["mitigated"]["stds"],
    marker="x",
    ms=ms,
    color=blue,
    linestyle="",
    lw=lw,
    mew=lw,
    capsize=cap,
    label=f"{backend_name} ZNE",
)

# Even
axs[1].errorbar(
    x,
    tenpy_evs[0][:num_obs],
    None,
    marker="x",
    ms=ms,
    color="k",
    linestyle="",
    lw=lw,
    mew=lw,
    capsize=cap,
    label="DMRG",
)
axs[1].errorbar(
    x,
    circuit_evs[0][:num_obs],
    None,
    marker="+",
    ms=ms,
    color="tab:green",
    linestyle="",
    lw=lw,
    mew=lw,
    capsize=cap,
    label="Compiled circuit:\nMPS simulation",
)
axs[1].errorbar(
    x,
    string_order["even"]["raw"]["evs"],
    string_order["even"]["raw"]["stds"],
    marker="x",
    ms=ms,
    color=red,
    linestyle="",
    lw=lw,
    mew=lw,
    capsize=cap,
    label=f"{backend_name} no ZNE",
)
axs[1].errorbar(
    x,
    string_order["even"]["mitigated"]["evs"],
    string_order["even"]["mitigated"]["stds"],
    marker="x",
    ms=ms,
    color=blue,
    linestyle="",
    lw=lw,
    mew=lw,
    capsize=cap,
    label=f"{backend_name} ZNE",
)

[axs[i].axhline(0, color="k") for i in range(2)]
[axs[i].axhline(1, color="k", linestyle="--") for i in range(2)]
# [axs[i].axhline(-1, color="k", linestyle="--") for i in range(2)]
axs[0].set_xlabel(r"String length, $l$")
[axs[i].set_xticks(x, x) for i in range(2)]
axs[0].set_title("Odd")
axs[1].set_xlabel(r"String length, $l$")
axs[1].set_title("Even")

axs[0].set_ylabel(
    r"$S^\mathrm{O}_{l,s} = (-1)^{l / 2} \langle \hat{Z}_{s+1} ... \hat{Z}_{s+l} \rangle$"
)
axs[1].set_ylabel(
    r"$S^\mathrm{E}_{l,s} = (-1)^{l / 2} \langle \hat{Z}_s ... \hat{Z}_{s+l-1} \rangle$"
)
axs[1].legend(loc="upper right")

# axs[0].set_box_aspect(1)
# axs[1].set_box_aspect(1)
fig.suptitle(
    f"$J_0={J0}, J_1={J1}$ string order,\n averaged over $s={str(s_list)[1:-1]}$"
)

fig.tight_layout()
fn = os.path.join(
    plot_dir,
    f"J0_{J0}_J1_{J1}_averaged_string_order_plots_s_{'_'.join([str(s) for s in s_list])}.pdf",
)
fig.savefig(
    fn,
    dpi=500,
)
print(f"Saved to: {fn}")
