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
import matplotlib.backends.backend_pdf
import numpy as np
import os
from datetime import datetime
from legs.bond_alternating_heisenberg_model.model import BondAlternatingModel
from legs.bond_alternating_heisenberg_model.utils import string_expectation_values
from matplotlib.figure import Figure
from pathlib import Path
from tenpy import MPS, TwoSiteDMRGEngine
from typing import List, Tuple

plt.rc("font", family="serif")
plt.rcParams.update({"font.size": 12})
plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

fig_dir = "figures/string_order/"
Path(fig_dir).mkdir(parents=True, exist_ok=True)

start_time = (
    str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_")[:-7]
)


def find_limit(x: List[float], tol: float = 1e-4) -> Tuple[float, Figure | None]:
    """
    Function to find the limit of values in a list. NOTE: if the values don't converge, this returns
    the mean of the final 10 values.

    Args:
        x (List[float]): The list of values.
        tol (float): The tolerance for convergence. If |x[i+1] - x[i]| < tol, returns x[i+1].
    Returns:
        The limit of the list, along with a plot of the convergence.
    """
    converged = False
    for i in range(len(x) - 1):
        if abs(x[i + 1] - x[i]) < tol:
            limit = x[i + 1]
            converged = True
            break

    if not converged:
        limit = np.mean(x[-10:])
        print(
            f"Warning: not converged, using the mean of the final 10 values ({limit})."
        )

    # Plots
    axis_vals = list(range(len(x)))
    fig, ax = plt.subplots()
    ax.scatter(axis_vals, x, marker="x")
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color="k", linestyle="-")
    if converged:
        # Plot where the sequence converged
        ax.axvline(axis_vals[i + 1], color="k", linestyle="--", label="Converged")
        ax.axhline(limit, color="r", label="Limit")
    else:
        # Plot the range averaged over
        ax.axvline(axis_vals[-10], color="k", linestyle="--")
        ax.axvline(
            axis_vals[-1], color="k", linestyle="--", label="Range averaged over"
        )
        ax.hlines(limit, axis_vals[-10], axis_vals[-1], color="r", label="Mean")

    ax.set_ylabel("Value")
    ax.set_xlabel("Index")
    ax.legend()

    return limit, fig


# Choose "fixed_J0" or "circle"
mode = "fixed_J0"

L = 100
steps = 21

# Set start_index >0 to avoid edge-effects. Should be even.
start_index = 20

if mode == "fixed_J0":
    J0 = 1.0
    J1_vals = np.linspace(-2.0, 2.0, steps)
    J_vals = [(J0, J1) for J1 in J1_vals]
    fn = f"L={L}_string_order_J0_{J0}_J1_min_{min(J1_vals)}_max_{max(J1_vals)}_{steps}_steps_starting_index_{start_index}_time_{start_time}.pdf"
elif mode == "circle":
    theta_vals = np.linspace(-np.pi, np.pi, steps)
    J_vals = [(np.cos(theta), np.sin(theta)) for theta in theta_vals]
    fn = f"L={L}_string_order_theta_-pi_to_pi_{steps}_steps_starting_index_{start_index}_time_{start_time}.pdf"
else:
    raise ValueError(f"Invalid mode: {mode}. Choose from 'fixed_J0' and 'circle'.")


# Save plots to PDF.
pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(fig_dir, fn))

# obs_list:
#   0: Even string order
#   1: Odd string order
#   2: Magnetisation
obs_list = [[], [], []]
for J0, J1 in J_vals:
    print(f"J0={J0}, J1={J1}")

    model = BondAlternatingModel(L, J0, J1, hz=1e-4)

    # Initialize a random MPS state as the starting point for DMRG
    psi = MPS.from_desired_bond_dimension(model.lat.mps_sites(), 1)

    # Initialise as a specific product state, can be useful for degenerate ground states.
    # string = ["up", "down"] * 25 + ["down", "up"] * 25
    # psi = MPS.from_product_state(model.lat.mps_sites(), string)

    # DMRG parameters
    trunc_thr = 1e-10
    dmrg_params = {
        "trunc_params": {
            "svd_min": 1e-10,
            "chi_max": 10,
            "trunc_cut": np.sqrt(trunc_thr),
        },
        "mixer": True,  # Add noise to escape local minima
        "max_sweeps": 20,
        "combine": True,  # Combine previous sweeps' truncation
    }

    # Run the DMRG algorithm to obtain the ground state
    dmrg_engine = TwoSiteDMRGEngine(psi, model, dmrg_params)
    E, psi = dmrg_engine.run()
    psi: MPS = psi

    obs_list[2].append(np.mean(psi.expectation_value("Sz")))

    # Calculate string expectation values for the string-order.
    even_values, odd_values = string_expectation_values(psi, start_index)

    # Find the limit of the even(odd) values. This is the string order parameter.
    for i in range(2):
        limit, fig = find_limit(even_values if i == 0 else odd_values)

        fig.suptitle(
            f"{'Even' if i == 0 else 'Odd'} string order convergence:\n$J_0 = {np.round(J0, 3)}$, $J_1 = {np.round(J1, 3)}$."
        )
        pdf.savefig(fig, dpi=500)
        plt.close(fig)

        obs_list[i].append(limit)

if mode == "fixed_J0":
    x = J1_vals
    x_label = f"$J_1$ ($J_0 = {J0}$)"
    v_line = 1  # J0=J1 line
    title = (
        "Order parameters for fixed $J_0$\n"
        + r"$O_{\text{str}}=\text{lim}_{j\rightarrow\infty}(-1)^{j+1}\langle Z_s Z_{s+1} \dots Z_{s+2j+1} \rangle$"
    )
elif mode == "circle":
    x = theta_vals
    x_label = r"$\theta$ ($J_0 = \text{cos}(\theta), J_1 = \text{sin}(\theta)$)"
    v_line = np.pi / 4  # J0=J1 line
    title = (
        "Order parameters around a circle\n"
        + r"$O_{\text{str}}=\text{lim}_{j\rightarrow\infty}(-1)^{j+1}\langle Z_s Z_{s+1} \dots Z_{s+2j+1} \rangle$"
    )

y_labels = [
    r"$O_{\text{str,even}}$",
    r"$O_{\text{str,odd}}$",
    r"$M=\frac{1}{L}\sum\limits_{i}{\langle S^z_i \rangle}$",
]
colours = ["r", "b", "g"]
labels = [
    f"`Even': $s={start_index}$",
    f"`Odd': $s={start_index + 1}$",
    "Magnetisation",
]

fig, ax = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
[
    ax[i].scatter(x, obs_list[i], marker="x", color=colours[i], label=labels[i])
    for i in range(3)
]
ax[0].set_ylim(-1.1, 1.1)
ax[1].set_ylim(-1.1, 1.1)
ax[2].set_ylim(-0.6, 0.6)
ax[2].set_xlabel(x_label)
[ax[i].set_ylabel(y_labels[i]) for i in range(3)]
[ax[i].axhline(0, color="k") for i in range(3)]
[ax[i].axvline(0, color="k") for i in range(3)]
[
    ax[i].axvline(
        v_line, color="k", linestyle="--", label=r"$J_0 = J_1$" if i == 0 else None
    )
    for i in range(3)
]
[ax[i].legend() for i in range(3)]
fig.suptitle(title)
fig.tight_layout()
pdf.savefig(fig, dpi=500)

pdf.close()
