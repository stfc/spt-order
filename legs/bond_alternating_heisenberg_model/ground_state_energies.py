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
import os
from datetime import datetime
from pathlib import Path
from legs.bond_alternating_heisenberg_model.model import BondAlternatingModel
from legs.bond_alternating_heisenberg_model.utils import load_mps_from_file


plt.rc("font", family="serif")
plt.rcParams.update({"font.size": 12})
plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

fig_dir = "figures/compiled_energies/"
Path(fig_dir).mkdir(parents=True, exist_ok=True)

start_time = (
    str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_")[:-7]
)


L = 100
J0 = 1.0

x_vals = []
y_vals = {"target": [], "AQC": [], "Ran_1": [], "Ran_10": []}

# Add file paths.
files = {
    0.5: {
        "target": "../bond_alternating_heisenberg_model/saved_mps/target_n_100_J0_1.0_J1_0.5_hz_0.0_trunc_1e-12_compressed_bd_5.pkl",
        "AQC": "../bond_alternating_heisenberg_model_compiling/results/n_100_layers_3_J0_1.0_J1_0.5_tt_1e-12_1756375695282178000_fidelity_0.9900.txt",
        "Ran": "../comparison_to_other_techniques/results_bahm/ran_method_compress_to_None_target_n_100_J0_1.0_J1_0.5_hz_0.0_trunc_1e-12_compressed_bd_5.json",
    },
    -1.0: {
        "target": "../bond_alternating_heisenberg_model/saved_mps/target_n_100_J0_1.0_J1_-1.0_hz_0.0_trunc_1e-12_compressed_bd_8.pkl",
        "AQC": "../bond_alternating_heisenberg_model_compiling/results/n_100_layers_3_J0_1.0_J1_-1.0_tt_1e-12_1756388302044458855_fidelity_0.9900.txt",
        "Ran": "../comparison_to_other_techniques/results_bahm/ran_method_compress_to_None_target_n_100_J0_1.0_J1_-1.0_hz_0.0_trunc_1e-12_compressed_bd_8.json",
    },
    -2.0: {
        "target": "../bond_alternating_heisenberg_model/saved_mps/target_n_100_J0_1.0_J1_-2.0_hz_0.0_trunc_1e-12_compressed_bd_8.pkl",
        "AQC": "../bond_alternating_heisenberg_model_compiling/results/n_100_layers_6_J0_1.0_J1_-2.0_tt_1e-12_1756388702662351325_fidelity_0.9794.txt",
        "Ran": "../comparison_to_other_techniques/results_bahm/ran_method_compress_to_None_target_n_100_J0_1.0_J1_-2.0_hz_0.0_trunc_1e-12_compressed_bd_8.json",
    },
}

fn = f"J0_{J0}_J1_{'_'.join([str(i) for i in sorted(files.keys())])}"

dmrg_E1 = []
if os.path.isfile("./energies.json"):
    with open("./energies.json", "r") as file:
        dmrg_energies = json.load(file)
else:
    raise ValueError(
        "Please generate a './energies.json' file by running './first_excited_state_dmrg.py'."
    )

for J1 in files.keys():
    model = BondAlternatingModel(100, J0, J1)
    H = model.calc_H_MPO()
    x_vals.append(J1)
    for method in files[J1].keys():
        if method == "Ran":
            for layers in [1, 10]:
                mps = load_mps_from_file(files[J1][method], layers)
                y_vals[f"Ran_{layers}"].append(np.real(H.expectation_value(mps)))
        else:
            mps = load_mps_from_file(files[J1][method])
            y_vals[method].append(np.real(H.expectation_value(mps)))

    dmrg_E1.append(dmrg_energies[str((L, J0, J1))]["E1"])

print(x_vals)
print(y_vals)

# Theoretical bounds. The eigenvalues of <S_i dot S_i+1> are -3/4 and +1/4. Therefore, the max(min)
# value that each J_i <S_i dot S_i+1> term can take is:
# (-1/4 +/- 1/2 * sign(J_i)) * J_i.
# E.g. max 1/4 J_i if J_i > 0, or -3/4 J_i if J_i < 0.

# If all of these terms are saturated, the minimum and maximum energies would be the sum of the
# above. NOTE: These are not tight bounds. I.e. the minimum and maximum energy states will probably
# not reach these energies, but they can't go beyond.
extended_x_vals = np.linspace(min(x_vals), max(x_vals))
maximum = (L // 2) * (
    -0.25 + 0.5 * np.sign(J0 * np.ones(extended_x_vals.shape))
) * J0 * np.ones(extended_x_vals.shape) + (L - 1 - L // 2) * (
    -0.25 + 0.5 * np.sign(extended_x_vals)
) * extended_x_vals
minimum = (L // 2) * (
    -0.25 - 0.5 * np.sign(J0 * np.ones(extended_x_vals.shape))
) * J0 * np.ones(extended_x_vals.shape) + (L - 1 - L // 2) * (
    -0.25 - 0.5 * np.sign(extended_x_vals)
) * extended_x_vals

for method in y_vals.keys():
    plt.scatter(x_vals, y_vals[method], marker="x", label=method, alpha=0.5)

plt.scatter(x_vals, dmrg_E1, color="k", marker="*", label="DMRG E1")

plt.axvline(0, color="k")
plt.xlabel(r"$J_1$")
plt.ylabel(r"$\langle H \rangle$")
plt.title(r"Compiled circuit energies for fixed $J_0=1.0$")

plt.legend()
plt.savefig(os.path.join(fig_dir, f"{fn}_compiled_energies.pdf"), dpi=100)


# Make a separate plot including the bounds.
plt.plot(
    extended_x_vals,
    maximum,
    color="k",
    linestyle="--",
    label="Loose theoretical bounds",
)
plt.plot(
    extended_x_vals,
    minimum,
    color="k",
    linestyle="--",
)

plt.axhline(0, color="k")
plt.legend()
plt.savefig(os.path.join(fig_dir, f"{fn}_compiled_energies_with_bounds.pdf"), dpi=100)
