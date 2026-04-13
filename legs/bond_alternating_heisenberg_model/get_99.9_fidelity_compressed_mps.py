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


import pickle
import os
from pathlib import Path
from adaptaqc.utils.utilityfunctions import qiskit_to_tenpy_mps
import copy
import matplotlib.pyplot as plt
from adaptaqc.utils.utilityfunctions import tenpy_to_qiskit_mps


# Load un-compressed MPS
target_fn = (
    "saved_mps/target_n_100_J0_1.0_J1_-1.0_hz_0.0_trunc_1e-12_compressed_bd_None.pkl"
)
with open(target_fn, "rb") as mps_f:
    raw_target_mps = pickle.load(mps_f)

tenpy_mps = qiskit_to_tenpy_mps(raw_target_mps)
original_tenpy_mps = copy.deepcopy(tenpy_mps)

print(max(original_tenpy_mps.chi))

# Compress to max bond dimension 1, 2, 3, ... until 99.9% fidelity is reached.
for max_bond in range(1, max(original_tenpy_mps.chi) + 1):

    tenpy_mps = qiskit_to_tenpy_mps(raw_target_mps)

    compression_options = {
        "compression_method": "variational",
        "trunc_params": {"chi_max": max_bond},
        "max_trunc_err": 1,
        "max_sweeps": 50,
        "min_sweeps": 10,
    }
    tenpy_mps.compress(compression_options)
    tenpy_mps.norm = 1

    fidelity = abs(tenpy_mps.overlap(original_tenpy_mps)) ** 2
    print(f"Max bond: {max(tenpy_mps.chi)}, fidelity: {fidelity}")

    if fidelity >= 0.999:
        qiskit_mps = tenpy_to_qiskit_mps(tenpy_mps)
        new_fn = target_fn.replace("None", f"{max_bond}")
        with open(new_fn, "wb") as f:
            pickle.dump(qiskit_mps, f)
        break

dir = "./figures/ground_state_properties/"
Path(dir).mkdir(parents=True, exist_ok=True)
fn = new_fn.replace("saved_mps/target_", "").replace(".pkl", "")

plt.plot(tenpy_mps.expectation_value("Sz"), label="Sz", marker="x")
plt.ylim(-0.6, 0.6)
plt.legend()
plt.savefig(os.path.join(dir, f"{fn}_sz.png"), dpi=100)
