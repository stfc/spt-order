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
import os
from legs.bond_alternating_heisenberg_model.model import BondAlternatingModel
from tenpy import MPS
from tenpy.algorithms import dmrg

# List of (l, J0, J1)
params_list = [
    (100, 1.0, 0.5),
    (100, 1.0, -1.0),
    (100, 1.0, -2.0),
    (100, 0.5, 1.0),
]

save_to = "./energies.json"

if os.path.isfile(save_to):
    with open(save_to, "r") as file:
        data = json.load(file)
    existing_keys = data.keys()
else:
    data = {}
    existing_keys = []

for params in params_list:
    if str(params) in existing_keys:
        print(f"Skipping {params}, already in file.")
        continue

    print(f"(l, J0, J1) = {params}")
    (l, J0, J1) = params
    model = BondAlternatingModel(l, J0, J1, hz=0)

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

    # Initialize a random MPS state as the starting point for DMRG
    if J0 < J1 and J1 > 0:
        print("Odd Haldane phase, four-fold degenerate ground state")
        product_states = [
            MPS.from_product_state(
                model.lat.mps_sites(), ["up", "down"] * 25 + ["down", "up"] * 25
            ),
            MPS.from_product_state(
                model.lat.mps_sites(), ["down", "up"] * 25 + ["up", "down"] * 25
            ),
            MPS.from_product_state(model.lat.mps_sites(), ["up", "down"] * 50),
            MPS.from_product_state(model.lat.mps_sites(), ["down", "up"] * 50),
        ]
        ground_states = []
        E0_list = []
        for product_state in product_states:

            # Run the DMRG algorithm to obtain the ground state
            dmrg_engine = dmrg.TwoSiteDMRGEngine(product_state, model, dmrg_params)
            E0, psi0 = dmrg_engine.run()
            ground_states.append(psi0)
            E0_list.append(E0)

            print("E0:", E0)
            print(
                f"Edge magnetisation: {np.round(np.sum(psi0.expectation_value('Sz')[:20]), 3)}, {np.round(np.sum(psi0.expectation_value('Sz')[-20:]), 3)}"
            )
    else:
        product_state = MPS.from_desired_bond_dimension(model.lat.mps_sites(), 1)
        # Run the DMRG algorithm to obtain the ground state
        dmrg_engine = dmrg.TwoSiteDMRGEngine(product_state, model, dmrg_params)
        E0, psi0 = dmrg_engine.run()
        ground_states = [psi0]
        E0_list = [E0]

        print("E0:", E0)

    # Run DMRG whilst orthogonal to the ground state to get the first excited state.
    psi1 = MPS.from_desired_bond_dimension(model.lat.mps_sites(), 1)
    dmrg_engine = dmrg.TwoSiteDMRGEngine(psi1, model, dmrg_params)
    dmrg_engine.init_env(model, orthogonal_to=ground_states)
    E1, _ = dmrg_engine.run()

    print("E1:", E1)
    for E0, psi0 in zip(E0_list, ground_states):
        print(f"Gap: {E1 - E0}")
        print(f"Overlap (should be 0): {psi0.overlap(psi1)}")
    print("")

    data[str(params)] = {"E0": E0_list if len(E0_list) > 1 else E0_list[0], "E1": E1}

with open(save_to, "w") as file:
    json.dump(data, file, indent=4)
