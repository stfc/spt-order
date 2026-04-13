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
from datetime import datetime
from pathlib import Path
from legs.bond_alternating_heisenberg_model.model import BondAlternatingModel
from legs.bond_alternating_heisenberg_model.utils import load_mps_from_file
import matplotlib.pyplot as plt


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
J1 = -2.0

dmrg = "../bond_alternating_heisenberg_model_compiling/saved_mps/target_n_100_J0_1.0_J1_-2.0_hz_0.0_trunc_1e-12_compressed_bd_None.pkl"
target = "../bond_alternating_heisenberg_model_compiling/saved_mps/target_n_100_J0_1.0_J1_-2.0_hz_0.0_trunc_1e-12_compressed_bd_8.pkl"
aqc = "../bond_alternating_heisenberg_model_compiling/results/n_100_layers_6_J0_1.0_J1_-2.0_tt_1e-12_1756388702662351325_fidelity_0.9794.txt"

assert (f"n_{L}" in dmrg) and (f"n_{L}" in target) and (f"n_{L}" in aqc)
assert (f"J0_{J0}" in dmrg) and (f"J0_{J0}" in target) and (f"J0_{J0}" in aqc)
assert (f"J1_{J1}" in dmrg) and (f"J1_{J1}" in target) and (f"J1_{J1}" in aqc)


model = BondAlternatingModel(100, J0, J1)
H = model.calc_H_MPO()
dmrg_mps = load_mps_from_file(dmrg)
print(f"DMRG chi: {max(dmrg_mps.chi)}")
print(f"DMRG energy: {np.real(H.expectation_value(dmrg_mps))}")

model = BondAlternatingModel(100, J0, J1)
H = model.calc_H_MPO()
target_mps = load_mps_from_file(target)
print(f"Target chi: {max(target_mps.chi)}")
print(f"Target energy: {np.real(H.expectation_value(target_mps))}")


model = BondAlternatingModel(100, J0, J1)
H = model.calc_H_MPO()
compiled_mps = load_mps_from_file(aqc)
print(f"Compiled energy: {np.real(H.expectation_value(compiled_mps))}")

print(np.abs(dmrg_mps.overlap(compiled_mps)) ** 2)
