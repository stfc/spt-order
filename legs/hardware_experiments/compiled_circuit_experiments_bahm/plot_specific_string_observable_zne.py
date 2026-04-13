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
import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from qiskit.quantum_info import SparsePauliOp
from tenpy import Array
from legs.hardware_experiments.utils import get_observable_results, functions
from legs.bond_alternating_heisenberg_model.utils import load_mps_from_file
from legs.hardware_experiments.compiled_circuit_experiments_bahm.constants import (
    EXPERIMENT_TO_PLOT,
)

plt.rc("font", family="serif")
plt.rc("text", usetex=True)
plt.rcParams["figure.figsize"] = 3, 3
plt.rcParams.update({"font.size": 16})

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
target_mps = load_mps_from_file(target_mps_fn)

# Compiled circuit
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

# Define observable to plot
# Define multiple s values to compute the string order starting at multiple sites.
obs_type = "even"
s = 60
l = 20
if obs_type == "even":
    obs = SparsePauliOp.from_sparse_list(
        [("Z" * l, list(range(s, s + l)), (-1) ** (l // 2))], n
    )
elif obs_type == "odd":
    obs = SparsePauliOp.from_sparse_list(
        [("Z" * l, list(range(s + 1, s + l + 1)), (-1) ** (l // 2))],
        n,
    )
else:
    raise ValueError(f"Unknown observable type: {obs_type}")

observable_name = f"{obs_type}_string_order_s_{s}_l_{l}"


obs_results = get_observable_results(obs, result_dict)


# Pauli Z in SpinSite basis convention.
op = Array.from_ndarray_trivial(
    np.array([[-1, 0], [0, 1]]),
    labels=["p", "p*"],
)

target_obs = target_mps.expectation_value_multi_sites(operators=[op] * l, i0=s) * (
    -1
) ** (l // 2)
circuit_obs = circuit_mps.expectation_value_multi_sites(operators=[op] * l, i0=s) * (
    -1
) ** (l // 2)


# Save figures in the right directory
split_fn = result_fn.split("/")
results_index = split_fn.index("results")
plot_dir = "/".join(split_fn[:results_index]) + "/figures/string_order_plots/"
Path(plot_dir).mkdir(parents=True, exist_ok=True)


noise_factors = result_dict["PrimitiveResult.metadata"]["resilience"]["zne"][
    "noise_factors"
]
extrapolated_noise_factors = result_dict["PrimitiveResult.metadata"]["resilience"][
    "zne"
]["extrapolated_noise_factors"]

ms = 8
lw = 1.5
cap = 3

# Classical references
plt.scatter(
    0,
    target_obs,
    color="k",
    marker="x",
    label="DMRG",
    s=ms**2,
)
plt.scatter(
    0,
    circuit_obs,
    color="tab:green",
    marker="+",
    label="Compiled circuit:\nMPS simulation",
    s=ms**2,
)

# Raw EVs at each noise factor.
evs_noise_factors = obs_results["evs_noise_factors"][0]
plt.errorbar(
    noise_factors,
    evs_noise_factors,
    obs_results["stds_noise_factors"][0],
    marker="x",
    color=red,
    linestyle="",
    ms=ms,
    lw=lw,
    mew=lw,
    capsize=cap,
    label=backend_name,
)
# Qiskit's extrapolation to 0 (will likely use different extrapolators for different Pauli strings).
plt.errorbar(
    0,
    obs_results["evs"][0],
    obs_results["stds"][0],
    marker="x",
    color=blue,
    linestyle="",
    ms=ms,
    lw=lw,
    mew=lw,
    capsize=cap,
)

chosen_extrapolator = obs_results["extrapolator"][0]
func = functions[chosen_extrapolator]
extrapolator_index = result_dict["PrimitiveResult.metadata"]["resilience"]["zne"][
    "extrapolator"
].index(chosen_extrapolator)
evs_extrapolated = obs_results["evs_extrapolated"][0][extrapolator_index]

x = np.linspace(0, max(noise_factors), 100)
# # My fit for the data
# popt, _ = curve_fit(func, noise_factors, evs_noise_factors)
# y = func(x, *popt)
# plt.plot(x, y, alpha=0.5, color="k", linestyle="--")

# Qiskit's fit
popt, _ = curve_fit(func, extrapolated_noise_factors, evs_extrapolated)
y = func(x, *popt)
plt.plot(x, y, alpha=1.0, label="Extrapolation", color=blue)

plt.xticks([0, 1, 2])
plt.axhline(0, color="k")
plt.axhline(1, color="k", linestyle="--")
plt.xlabel("Noise factor")
plt.ylabel(r"$\langle \hat{O} \rangle$")
# plt.title(f"$J_0={J0}, J_1={J1}$")
# plt.legend(loc="upper right", fontsize=10)
plt.tight_layout()
fn = os.path.join(plot_dir, f"{observable_name}_ZNE.pdf")
plt.savefig(fn, dpi=500)
print(f"Saved to: {fn}")
