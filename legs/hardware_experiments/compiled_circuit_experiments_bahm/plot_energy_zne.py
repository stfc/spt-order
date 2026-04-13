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
from qiskit.quantum_info import SparsePauliOp
from legs.hardware_experiments.utils import get_observable_results
from legs.hardware_experiments.compiled_circuit_experiments_bahm.constants import (
    EXPERIMENT_TO_PLOT,
)

plt.rc("font", family="serif")
plt.rc("text", usetex=True)

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


# Hardware results
result_fn = experiments[str(EXPERIMENT_TO_PLOT)]["result_fn"]
assert f"J0_{J0}" in result_fn
assert f"J1_{J1}" in result_fn
with open(result_fn, "rb") as file:
    result_dict = pkl.load(file)

# BAHM Hamiltonian: J_{i%2}(XX + YY + ZZ)_{i, i+1}
hamiltonian_term_list = []
for i in range(n - 1):
    for term in ["XX", "YY", "ZZ"]:
        hamiltonian_term_list.append(
            (term, [i, i + 1], J0 / 4 if i % 2 == 0 else J1 / 4)
        )
hamiltonian_observable = [SparsePauliOp.from_sparse_list(hamiltonian_term_list, n)]

energy_results = get_observable_results(hamiltonian_observable, result_dict)


# Save figures in the right directory
split_fn = result_fn.split("/")
results_index = split_fn.index("results")
plot_dir = "/".join(split_fn[:results_index]) + "/figures/energy_plots"
Path(plot_dir).mkdir(parents=True, exist_ok=True)


noise_factors = result_dict["PrimitiveResult.metadata"]["resilience"]["zne"][
    "noise_factors"
]
extrapolated_noise_factors = result_dict["PrimitiveResult.metadata"]["resilience"][
    "zne"
]["extrapolated_noise_factors"]


exp_index = result_dict["PrimitiveResult.metadata"]["resilience"]["zne"][
    "extrapolator"
].index("exponential")
lin_index = result_dict["PrimitiveResult.metadata"]["resilience"]["zne"][
    "extrapolator"
].index("linear")

# Raw EVs at each noise factor.
plt.errorbar(
    noise_factors,
    energy_results["evs_noise_factors"][0],
    energy_results["stds_noise_factors"][0],
    marker="x",
    color="tab:blue",
    linestyle="",
    capsize=3,
    label="Raw EVs",
)
# Qiskit's extrapolation to 0 (will likely use different extrapolators for different Pauli strings).
plt.errorbar(
    0,
    energy_results["evs"][0],
    energy_results["stds"][0],
    marker="^",
    color="tab:blue",
    linestyle="",
    capsize=3,
    label="Qiskit extrapolation",
)
# Forced exponential fit.
plt.errorbar(
    extrapolated_noise_factors,
    energy_results["evs_extrapolated"][0, exp_index],
    energy_results["stds_extrapolated"][0, exp_index],
    marker="v",
    color="tab:orange",
    linestyle="",
    capsize=3,
    label="Exponential extrapolation",
)
# Forced linear fit.
plt.errorbar(
    extrapolated_noise_factors,
    energy_results["evs_extrapolated"][0, lin_index],
    energy_results["stds_extrapolated"][0, lin_index],
    marker="*",
    color="tab:green",
    linestyle="",
    capsize=3,
    label="Linear extrapolation",
)


# Add DMRG ground and first excited state energies.
if os.path.isfile("../../bond_alternating_heisenberg_model/energies.json"):
    with open("../../bond_alternating_heisenberg_model/energies.json", "r") as file:
        dmrg_energies = json.load(file)
else:
    raise ValueError(
        "Please generate a '../../bond_alternating_heisenberg_model/energies.json' file by running '../../bond_alternating_heisenberg_model/first_excited_state_dmrg.py'."
    )
plt.axhline(
    dmrg_energies[str((n, J0, J1))]["E0"], color="k", linestyle="--", label="DMRG E0"
)
plt.axhline(
    dmrg_energies[str((n, J0, J1))]["E1"], color="r", linestyle="--", label="DMRG E1"
)
plt.xlabel("Noise factor")
plt.ylabel(r"$\langle H \rangle$")
plt.title(f"$J_0={J0}, J_1={J1}$ energy ZNE")
plt.legend()
plt.savefig(os.path.join(plot_dir, "energy_zne.pdf"), dpi=500)
