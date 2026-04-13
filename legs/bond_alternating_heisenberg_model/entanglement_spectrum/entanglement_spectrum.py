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


import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from matplotlib import pyplot as plt
from legs.bond_alternating_heisenberg_model.utils import load_mps_from_file

plt.rc("font", family="serif")
plt.rcParams.update({"font.size": 10})
plt.rc("text", usetex=True)

dir = "./figures/"
Path(dir).mkdir(parents=True, exist_ok=True)

# If the ground state circuit is a Ran method circuit, you must specify the number of ladder layers.
ran_layers = 1

l = 100  # Length of the spin chain.
J0 = 1.0  # Even-odd bond strength.
J1 = -2.0  # Odd-even bond strength.
hz = 0.0  # External magnetic field.
trunc = 1e-12
compressed_bd = 8

# Filepath to the saved MPS. This can be a .pkl file with an MPS, a .txt file containing a QASM
# circuit, or a .json file from schon_and_ran_method.py.
mps_fn = (
    "../saved_mps/target_n_100_J0_1.0_J1_-2.0_hz_0.0_trunc_1e-12_compressed_bd_8.pkl"
)

assert f"n_{l}" in mps_fn
assert f"J0_{J0}" in mps_fn
assert f"J1_{J1}" in mps_fn

mps = load_mps_from_file(mps_fn, ran_layers)

if mps_fn.endswith(".pkl"):
    file_type = "Target MPS"
elif mps_fn.endswith(".txt"):
    file_type = "Compiled circuit"
elif mps_fn.endswith(".json"):
    file_type = f"Ran method: L = {ran_layers}"


# Extract the `num_evals` largest eigenvalues of the RDM.
num_evals = 25

# Compute the RDMs of 1 to `max_length` sites from the end of the MPS.
max_length = 10

# Discard eigenvalues less than tol.
tol = 1e-10

fig, axs = plt.subplots(1, 2, sharey=True)
offset = [0, 0]

# Ticks and labels: [even l, odd l]
x_ticks = [[], []]
x_labels = [[], []]
for l in list(range(1, max_length + 1)):
    rdm = mps.get_rho_segment(list(range(l)))
    array = rdm.to_ndarray().reshape((2**l, 2**l))

    # Largest `num_evals` eigenvalues in descending order
    evals = np.linalg.eigh(array)[0][::-1][:num_evals]

    x_ticks[l % 2].append(offset[l % 2])
    x_labels[l % 2].append(l)

    if l > 2:
        axs[(l + 1) % 2].axvline(
            offset[l % 2] - 1, linestyle="--", color="k", linewidth=0.5, alpha=0.7
        )

    for i in range(evals.shape[0]):
        if evals[i] < tol:
            break

        if l == 1 and i == 0:
            label = "J0 cut"
        elif l == 2 and i == 0:
            label = "J1 cut"
        else:
            label = None

        axs[(l + 1) % 2].bar(
            offset[l % 2],
            evals[i],
            color="tab:blue" if l % 2 == 1 else "tab:orange",
            label=label,
        )
        offset[l % 2] += 1

    offset[l % 2] += 1

axs[0].set_yscale("log")
axs[1].set_yscale("log")


plt.suptitle(f"Entanglement Spectrum\n$J_0={J0}$, $J_1={J1}$, {file_type}")
# Odd l goes in axs[0].
axs[0].set_title(r"Odd-length segments - $J_0$ cuts")
axs[1].set_title(r"Even-length segments - $J_1$ cuts")
axs[0].set_xlabel(r"Segment length ($l$)")
axs[1].set_xlabel(r"Segment length ($l$)")
axs[0].set_ylabel(
    "Largest eigenvalues of the reduced density\nmatrix for sites 0 to "
    + r"$l-1$ inclusive"
)
# Odd l goes in axs[0], hence slightly confusing indices...
axs[0].set_xticks(x_ticks[1])
axs[1].set_xticks(x_ticks[0])
axs[0].set_xticklabels(x_labels[1])
axs[1].set_xticklabels(x_labels[0])

fig.tight_layout()
fn = mps_fn.split("/")[-1].replace(".txt", "").replace(".pkl", "").replace(".json", "")
path = os.path.join(dir, f"{fn}_entanglement_spectrum.pdf")
plt.savefig(path, dpi=500)
print(f"Plot saved to {path}")
