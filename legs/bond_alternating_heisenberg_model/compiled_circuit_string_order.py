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
import pickle as pkl
import os
from datetime import datetime
from pathlib import Path
from adaptaqc.utils.utilityfunctions import qiskit_to_tenpy_mps
from scipy.optimize import curve_fit
from legs.bond_alternating_heisenberg_model.utils import (
    string_expectation_values,
    load_mps_from_file,
)
from textwrap import wrap


def exponential(x, a, b, c):
    return a * np.exp(-x / b) + c


plt.rc("font", family="serif")
plt.rcParams.update({"font.size": 12})
plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

fig_dir = "figures/compiled_string_order/"
Path(fig_dir).mkdir(parents=True, exist_ok=True)

start_time = (
    str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_")[:-7]
)

L = 100
J0 = 1.0

# If any compiled circuit files are Ran method .json files, use this many ladder layers.
ran_layers = 1

# List of:
#   - J1 value.
#   - Compiled circuit file name. Can either be a .txt QASM file or one of the Schon/Ran .json files.
#   - The corresponding target MPS file name.
# For each element of the list, a plot will be generated showing the individual even/odd string-
# order expectation values for the compiled circuit and the target MPS, along with a .json file
# containing fitting parameters. You can find a list of relevant file-paths in file_paths.txt.
combinations = [
    (
        0.5,
        "../bond_alternating_heisenberg_model_compiling/results/n_100_layers_3_J0_1.0_J1_0.5_tt_1e-12_1756375695282178000_fidelity_0.9900.txt",
        "../bond_alternating_heisenberg_model/saved_mps/target_n_100_J0_1.0_J1_0.5_hz_0.0_trunc_1e-12_compressed_bd_5.pkl",
    ),
]

for J1, compiled_circuit_fn, target_mps_fn in combinations:
    # Load ground state circuit
    assert f"J1_{J1}" in compiled_circuit_fn

    circuit_mps = load_mps_from_file(compiled_circuit_fn, ran_layers)
    # AQC-compiled circuit.
    if compiled_circuit_fn.endswith(".txt"):
        fn = compiled_circuit_fn.split("/")[-1].removesuffix(".txt")
    # Ran metrics JSON
    elif compiled_circuit_fn.endswith(".json"):
        fn = f"layers_{ran_layers}_{compiled_circuit_fn.split("/")[-1].removesuffix(".json")}"
    else:
        raise ValueError("Unknown file format.")

    # Target MPS
    assert f"J1_{J1}" in target_mps_fn
    with open(target_mps_fn, "rb") as mps_f:
        raw_target_mps = pkl.load(mps_f)

    target_mps = qiskit_to_tenpy_mps(raw_target_mps)

    # Set start_index >0 to avoid edge-effects. Should be even.
    start_index = 20

    # Compute -<ZZ>, <ZZZZ>, ... expectation values.
    circuit_evs = string_expectation_values(circuit_mps, start_index)
    target_evs = string_expectation_values(target_mps, start_index)

    fig, axs = plt.subplots(1, 2)

    x = np.linspace(0, len(circuit_evs[0]) + 10, 100)

    results_dict = {"even": {}, "odd": {}}

    for i in range(4):
        index = i // 2
        if i % 2 == 0:
            evs = circuit_evs
            label = "Compiled circuit"
            colour = "tab:blue"
        else:
            evs = target_evs
            label = "Target MPS"
            colour = "tab:orange"

        # Plot values
        axs[index].scatter(
            list(range(len(evs[index]))),
            evs[index],
            marker="x",
            color=colour,
            label=label,
        )

        # Fit exponential: a * exp(-x / b) + c
        popt, pcov = curve_fit(
            exponential,
            list(range(len(evs[index]))),
            np.abs(evs[index]),
            bounds=([0, 0, -np.inf], [np.inf, L, np.inf]),
        )

        results_dict["even" if index == 0 else "odd"][label] = {
            "A": np.format_float_scientific(popt[0], 2),
            "A_err": np.format_float_scientific(np.sqrt(pcov[0, 0]), 2),
            "B": np.format_float_scientific(popt[1], 2),
            "B_err": np.format_float_scientific(np.sqrt(pcov[1, 1]), 2),
            "C": np.format_float_scientific(popt[2], 2),
            "C_err": np.format_float_scientific(np.sqrt(pcov[2, 2]), 2),
            "condition_number": np.format_float_scientific(np.linalg.cond(pcov), 2),
        }

        axs[index].plot(x, exponential(x, popt[0], popt[1], popt[2]), color=colour)
        axs[index].errorbar(
            x[-1],
            popt[2],
            np.sqrt(pcov[2, 2]),
            marker="o",
            color=colour,
            alpha=0.5,
            label="Fit limit (inf)",
        )

    axs[0].set_ylim(-1.1, 1.1)
    axs[1].set_ylim(-1.1, 1.1)
    axs[0].axhline(0, color="k", linestyle="-")
    axs[1].axhline(0, color="k", linestyle="-")

    axs[0].set_title("Even string EVs")
    axs[1].set_title("Odd string EVs")
    axs[1].legend()

    axs[0].set_xlabel("Index: i")
    axs[0].set_ylabel(
        r"$(-1)^{i+1} \langle Z_s ... Z_{s+2i+1} \rangle, s=20(e), 21(o)$"
    )
    axs[1].set_xlabel("Index: i")

    fig.suptitle("\n".join(wrap(fn, width=50)))
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{fn}_plot.pdf"), dpi=100)

    with open(os.path.join(fig_dir, f"{fn}_params.json"), "w") as file:
        json.dump(results_dict, file, indent=4)
