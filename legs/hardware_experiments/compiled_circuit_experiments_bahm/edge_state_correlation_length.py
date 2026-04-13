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
import os
import numpy as np
from pathlib import Path
from legs.bond_alternating_heisenberg_model.utils import (
    load_mps_from_file,
)
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
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
plt.rcParams.update({"font.size": 12})
plt.rc("text", usetex=True)


def exponential(x, a, b):
    return a * np.exp(-x / b)


red = "#FF644E"
blue = "#00A2FF"

n = 100
J0 = 1.0
J1 = -2.0

# If `num_sites` is not None, two separate panels will be plotted, showing <Sz> for `num_sites` from
# each end of the chain
num_sites = 20

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
plot_dir = "/".join(split_fn[:results_index]) + "/figures/sz_plots"
Path(plot_dir).mkdir(parents=True, exist_ok=True)

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

raw_evs = z_results["evs_noise_factors"][:, 0] / 2
raw_stds = z_results["stds_noise_factors"][:, 0] / 2

# figsize=(10, 2.5)
fig, axs = plt.subplots(1, 4, figsize=(8, 3.2), sharey=True)

evs = [tenpy_sz, circuit_sz, raw_evs, extrapolated_evs]
stds = [None, None, raw_stds, extrapolated_stds]

colours = ["k", "tab:green", red, blue]
labels = [
    "DMRG",
    "Compiled circuit:\nMPS simulation",
    f"{backend_name}:\nno ZNE",
    f"{backend_name}:\nlinear ZNE",
]

cap = 2

for i in range(4):
    # Block the sites in groups of 2 and sum the magnetisation.
    ev = evs[i][:num_sites]
    std = stds[i] if stds[i] is None else stds[i][:num_sites]

    ev = [ev[2 * i] + ev[2 * i + 1] for i in range(num_sites // 2)]
    if std is not None:
        std = [
            np.sqrt(std[2 * i] ** 2 + std[2 * i + 1] ** 2)
            for i in range(num_sites // 2)
        ]

    sites = list(range(num_sites // 2))

    popt, pcov = curve_fit(exponential, sites, ev, sigma=std)

    xi = 2 * popt[1]
    xi_err = 2 * np.sqrt(pcov[1, 1])
    gap = 1 / xi
    gap_err = xi_err / xi**2

    print("Corr. len.")
    print(f"xi = {xi} +/- {xi_err}")
    print("Energy gap")
    print(f"gap = {gap} +/- {gap_err}")
    print("")

    # row = i // 2
    # col = i % 2

    axs[i].axhline(0, color="k", linestyle="-")
    axs[i].errorbar(
        sites,
        ev,
        std,
        linestyle="",
        marker="x",
        color=colours[i],
        label=labels[i],
        capsize=cap,
    )
    x = np.linspace(0, num_sites // 2 - 1, 100)
    axs[i].plot(
        x,
        exponential(x, *popt),
        color=colours[i],
        label=f"$\\xi={np.round(xi, 3)}\\pm{np.round(xi_err, 3)}$\n$\\Delta = {np.round(gap, 3)}\\pm {np.round(gap_err, 3)}$",
    )
    axs[i].legend(prop={"size": 8})
    axs[i].set_box_aspect(1.5)

axs[0].set_ylabel(
    r"$(\langle \hat{Z}_{2i} \rangle + \langle \hat{Z}_{2i+1} \rangle)/2$"
)
fig.supxlabel(r"Unit cell index, $i$")

plt.tight_layout()
fn = os.path.join(plot_dir, "edge_state_correlation_length.pdf")
plt.savefig(fn, dpi=500)
print(f"Saved to: {fn}")
