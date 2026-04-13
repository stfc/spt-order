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


import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from legs.bond_alternating_heisenberg_model.utils import load_mps_from_file

plt.rc("font", family="serif")
plt.rc("text", usetex=True)
plt.rcParams.update({"font.size": 10})

red = "#FF644E"
blue = "#00A2FF"

n = 100
J0 = 1.0
J1 = -1.0
# Segment length.
l_list = list(range(1, 11))

# Extract the `num_evals` largest eigenvalues of the RDM.
num_evals = 16

# Discard eigenvalues less than tol.
tol = 1e-10


# Filepath to the saved MPS. This can be a .pkl file with an MPS, a .txt file containing a QASM
# circuit, or a .json file from schon_and_ran_method.py.
target_mps_fn = (
    "../saved_mps/target_n_100_J0_1.0_J1_-1.0_hz_0.0_trunc_1e-12_compressed_bd_None.pkl"
)
# "../../bond_alternating_heisenberg_model_compiling/results/n_100_layers_3_J0_0.5_J1_1.0_tt_1e-12_1769679224814412000_fidelity_0.9900.txt"
# "../../bond_alternating_heisenberg_model_compiling/results/n_100_layers_3_J0_1.0_J1_0.5_tt_1e-12_1756375695282178000_fidelity_0.9900.txt"
# "../../bond_alternating_heisenberg_model_compiling/results/n_100_layers_3_J0_1.0_J1_-1.0_tt_1e-12_1756388302044458855_fidelity_0.9900.txt"
# "../../bond_alternating_heisenberg_model_compiling/results/n_100_layers_6_J0_1.0_J1_-2.0_tt_1e-12_1756388702662351325_fidelity_0.9794.txt"
compiled_circuit_fn = "../../bond_alternating_heisenberg_model_compiling/results/n_100_layers_3_J0_1.0_J1_-1.0_tt_1e-12_1756388302044458855_fidelity_0.9900.txt"

assert f"n_{n}" in target_mps_fn
assert f"J0_{J0}" in target_mps_fn
assert f"J1_{J1}" in target_mps_fn
assert f"n_{n}" in compiled_circuit_fn
assert f"J0_{J0}" in compiled_circuit_fn
assert f"J1_{J1}" in compiled_circuit_fn

target_mps = load_mps_from_file(target_mps_fn)
compiled_mps = load_mps_from_file(compiled_circuit_fn)

print(max(target_mps.chi))


# Save figures in the right directory
plot_dir = "./figures/"
Path(plot_dir).mkdir(parents=True, exist_ok=True)


# TODO: Remove hard-coded numbers
fig, axs = plt.subplots(2, 5, sharey=True, constrained_layout=True)

for l in l_list:
    row = (l - 1) % 2
    col = (l - 1) // 2
    # Target MPS
    target_rdm = target_mps.get_rho_segment(list(range(l)))
    target_array = target_rdm.to_ndarray().reshape((2**l, 2**l))

    # Largest `num_evals` eigenvalues in descending order
    target_evals = np.linalg.eigh(target_array)[0][::-1][:num_evals]
    target_evals = target_evals[target_evals > tol]

    # Compiled circuit
    compiled_rdm = compiled_mps.get_rho_segment(list(range(l)))
    compiled_array = compiled_rdm.to_ndarray().reshape((2**l, 2**l))

    # Largest `num_evals` eigenvalues in descending order
    compiled_evals = np.linalg.eigh(compiled_array)[0][::-1][:num_evals]
    compiled_evals = compiled_evals[compiled_evals > tol]

    if row == 0 and col == 0:
        label = f"Compiled circuit, odd $l$"
    elif row == 1 and col == 0:
        label = f"Compiled circuit, even $l$"
    else:
        label = None

    axs[row, col].bar(
        list(range(1, compiled_evals.shape[0] + 1)),
        height=compiled_evals,
        color=red if row == 0 else blue,
        label=label,
    )

    axs[row, col].scatter(
        list(range(1, target_evals.shape[0] + 1)),
        target_evals,
        color="k",
        marker="x",
        label="DMRG" if row == 0 and col == 0 else None,
    )

    maximum = max(compiled_evals.shape[0], target_evals.shape[0])
    x = [1, maximum]
    x_labels = [f"$\lambda_{'{' + str(i) + '}'}$" for i in x]
    axs[row, col].set_xticks(x, x_labels)
    axs[row, col].set_title(f"$l=$ {l}")
    axs[row, col].set_yscale("log")
    # axs[row, col].axhline(0, color="k", linestyle="-")

fig.suptitle(f"Entanglement spectrum: $J_0={J0}$, $J_1={J1}$")
fig.legend(fontsize=8, loc="outside lower center", ncol=3)
# NOTE: I can't find a way to add an x axis label without it overlapping with the legend
# fig.supxlabel(r"$i^\mathrm{th}$ largest eigenvalue")

fn = os.path.join(
    plot_dir,
    f"n_{n}_J0_{J0}_J1_{J1}_dmrg_vs_aqc_entanglement_spectrum.pdf",
)
# fig.tight_layout()
fig.savefig(fn, dpi=500)
print(f"Figure saved to: {fn}")
