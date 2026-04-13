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


from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
from tenpy import MPS
import time
from legs.bond_alternating_heisenberg_model.utils import load_mps_from_file

plt.rc("font", family="serif")
plt.rcParams.update({"font.size": 8})
plt.rc("text", usetex=True)

dir = "./figures/correlations/"
Path(dir).mkdir(parents=True, exist_ok=True)


def correlation_function(mps: MPS):
    correlators = np.empty((mps.L, mps.L))
    correlators[:] = np.nan
    z = mps.expectation_value("Sz")
    for i in range(mps.L):
        for j in range(i + 1, mps.L):
            zizj = mps.expectation_value_term([("Sz", i), ("Sz", j)])
            correlator = zizj - z[i] * z[j]

            correlators[i, j] = correlator
            correlators[j, i] = correlator

    return correlators


# Set to True to load a compiled ground state circuit.
# NOTE: in this case, you must manually specify the file path.
ground_state_circuit = True
# If the ground state circuit is a Ran method circuit, you must specify the number of ladder layers.
ran_layers = 1

l = 100  # Length of the spin chain.
J0 = 1.0  # Even-odd bond strength.
J1 = 0.5  # Odd-even bond strength.
hz = 0.0  # External magnetic field.
trunc = 1e-12
compressed_bd = 5

if ground_state_circuit:
    ground_state_file = "../bond_alternating_heisenberg_model_compiling/results/n_100_layers_3_J0_1.0_J1_0.5_tt_1e-12_1756375695282178000_fidelity_0.9900.txt"
else:
    ground_state_file = f"./saved_mps/target_n_{l}_J0_{J0}_J1_{J1}_hz_{hz}_trunc_{trunc}_compressed_bd_{compressed_bd}.pkl"

assert f"n_{l}" in ground_state_file
assert f"J0_{J0}" in ground_state_file
assert f"J1_{J1}" in ground_state_file

tenpy_mps = load_mps_from_file(ground_state_file, ran_layers)

if ground_state_file.endswith(".pkl"):
    plot_fn = f"{dir}l_{l}_J0_{J0}_J1_{J1}_hz_{hz}_trunc_{trunc}_compressed_bd_{compressed_bd}_correlation_{time.time()}.png"
elif ground_state_file.endswith(".txt"):
    stripped_fn = ground_state_file.split("/")[-1].replace("txt", "png")
    plot_fn = f"{dir}circuit_TEST_{stripped_fn}"
elif ground_state_file.endswith(".json"):
    stripped_fn = ground_state_file.split("/")[-1].replace("json", "png")
    plot_fn = f"{dir}layers_{ran_layers}_TEST_{stripped_fn}"

# Ground state correlators
ground_state_correlators = correlation_function(tenpy_mps)

# Good scale
minimum = np.nanmin(ground_state_correlators)
maximum = np.nanmax(ground_state_correlators)

if abs(minimum) > abs(maximum):
    vmin = minimum
    vmax = -minimum
else:
    vmin = -maximum
    vmax = maximum

norm = SymLogNorm(linthresh=1e-3, vmin=vmin, vmax=vmax)

# Individual DMRG plot
extent = [0.5, l + 0.5, l + 0.5, 0.5]
fig = plt.imshow(ground_state_correlators, cmap="RdBu", norm=norm, extent=extent)

plt.colorbar(fig, orientation="vertical", label=r"$C_{zz}(i,j)$")
plt.xlabel(r"Site index: $i$")
plt.ylabel(r"Site index: $j$")
plt.title(r"MPS target: $|\psi_t\rangle$")

plt.savefig(
    plot_fn,
    dpi=500,
    bbox_inches="tight",
)
plt.clf()
