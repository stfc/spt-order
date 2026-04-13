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
import pickle
import numpy as np
from adaptaqc.utils.entanglement_measures import concurrence
from adaptaqc.utils.utilityfunctions import tenpy_to_qiskit_mps
from legs.bond_alternating_heisenberg_model.model import BondAlternatingModel
from matplotlib import pyplot as plt
from pathlib import Path
from tenpy import MPS
from tenpy.algorithms import dmrg

l = 100  # Length of the spin chain.
J0 = 1.0  # Even-odd bond strength.
J1 = -1.0  # Odd-even bond strength.
hz = 0.0  # External magnetic field.

model = BondAlternatingModel(l, J0, J1, hz)

# Initialize a random MPS state as the starting point for DMRG
psi = MPS.from_desired_bond_dimension(model.lat.mps_sites(), 1)

# Initialise as a specific product state, can be useful for degenerate ground states
# string = ["up", "down"] * 25 + ["down", "up"] * 25
# psi = MPS.from_product_state(model.lat.mps_sites(), string)

# DMRG parameters
trunc_thr = 1e-12
dmrg_params = {
    "trunc_params": {
        "svd_min": 1e-10,
        "chi_max": 100,
        "trunc_cut": np.sqrt(trunc_thr),
    },
    "mixer": True,  # Add noise to escape local minima
    "max_sweeps": 20,
    "combine": True,  # Combine previous sweeps' truncation
}

# Run the DMRG algorithm to obtain the ground state
dmrg_engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)
E, psi = dmrg_engine.run()
psi: MPS = psi

compress_to = None
if compress_to is not None:
    compression_options = {
        "compression_method": "variational",
        "trunc_params": {"chi_max": compress_to},
        "max_trunc_err": 1,
        "max_sweeps": 50,
        "min_sweeps": 10,
    }
    psi.compress(compression_options)
    psi.norm = 1


print("Energy:", E)
final_trunc_error = dmrg_engine.sweep_stats["max_trunc_err"][-1]
print("Final max trunc err", final_trunc_error)
final_delta_e = dmrg_engine.sweep_stats["Delta_E"][-1]
print("Final Delta_E", final_delta_e)
print("Final max χ", max(psi.chi))
print(psi.expectation_value("Sz")[0])

dir = "./figures/ground_state_properties/"
Path(dir).mkdir(parents=True, exist_ok=True)
fn = f"{dir}L_{l}_J0_{J0}_J1_{J1}_hz_{hz}_E_{E}_trunc_thr_{trunc_thr}_compressed_bd_{compress_to}"

e_vals = psi.expectation_value("Sz")
plt.plot(psi.expectation_value("Sz"), label="Sz", marker="x")
plt.ylim(-0.6, 0.6)
plt.title(
    f"E={np.round(E, 3)}, l_20 mag: {np.round(np.sum(e_vals[:20]), 3)}, r_20 mag: {np.round(np.sum(e_vals[-20:]), 3)}"
)
plt.legend()
plt.savefig(fn + "_tenpy_sz.png", dpi=100)
plt.clf()

plt.plot(psi.entanglement_entropy(), label="S")
plt.ylabel("Entanglement entropy S")
plt.ylim(0, 10)
plt.twinx()
plt.plot(psi.chi, color="orange", label=r"$\chi$")
plt.ylabel(r"Bond dimension $\chi$")
plt.ylim(0, 50)
plt.legend()
plt.savefig(fn + "_tenpy_bond_dim.png", dpi=100)
plt.clf()

rhos = [
    psi.get_rho_segment([i, i + 1]).to_ndarray().reshape(4, 4) for i in range(l - 1)
]
plt.plot([concurrence(rho) for rho in rhos], label="Concurrence")
plt.ylim(0, 1)
plt.legend()
plt.savefig(fn + "_tenpy_concurrence.png", dpi=100)

dir = "./saved_mps/"
fn = f"target_n_{l}_J0_{J0}_J1_{J1}_hz_{hz}_trunc_{trunc_thr}_compressed_bd_{compress_to}.pkl"
Path(dir).mkdir(parents=True, exist_ok=True)
qiskit_mps = tenpy_to_qiskit_mps(psi)
with open(os.path.join(dir, fn), "wb") as f:
    pickle.dump(qiskit_mps, f)
