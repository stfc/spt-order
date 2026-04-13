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


import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

plt.rc("font", family="serif")
plt.rcParams.update({"font.size": 12})
plt.rc("text", usetex=True)

red = "#FF644E"
blue = "#00A2FF"


# Curve fitting
def exponential(x, a, b):
    return a * np.exp(-x / b)


def linear(x, a, b):
    return a + b * x


def polynomial_degree_2(x, a, b, c):
    return a + b * x + c * x**2


functions = {
    "exponential": exponential,
    "linear": linear,
    "polynomial_degree_2": polynomial_degree_2,
}


# Results fn
result_fn = "../compiled_circuit_experiments_bahm/experiments/n_100_J0_1.0_J1_-2.0_layers_6/2025_11_18_09_31_14/results/results_ibm_fez_d4e3riheg65s738m1g90_identity_circuit.pkl"

with open(result_fn, "rb") as file:
    result_dict = pkl.load(file)

# Save figures in the right directory
split_fn = result_fn.split("/")
results_index = split_fn.index("results")
plot_dir = "/".join(split_fn[:results_index]) + "/figures/identity_zne"
Path(plot_dir).mkdir(parents=True, exist_ok=True)


# Magnetisation. Should be 1.
evs = result_dict["PubResult.data"]["evs"]
stds = result_dict["PubResult.data"]["stds"]

mag = sum(evs) / 100
err = np.sqrt(np.sum(np.square(stds))) / 100
print(f"Magnetisation: {mag}")
print(f"Error bar: {err}")

# fig_0: collated plot, fig_1: individual plots.
fig_0, axs_0 = plt.subplots(1, 2)
fig_0.set_figheight(5)
fig_0.set_figwidth(10)
fig_1, axs_1 = plt.subplots(10, 10)
fig_1.set_figheight(20)
fig_1.set_figwidth(20)

for qubit in range(100):
    row = qubit // 10
    col = qubit % 10

    # Raw data
    noise_factors = result_dict["PrimitiveResult.metadata"]["resilience"]["zne"][
        "noise_factors"
    ]
    evs_noise_factors = result_dict["PubResult.data"]["evs_noise_factors"][qubit]
    stds_noise_factors = result_dict["PubResult.data"]["stds_noise_factors"][qubit]

    chosen_extrapolator = result_dict["PubResult.metadata"]["resilience"]["zne"][
        "extrapolator"
    ][qubit]

    try:
        x = np.linspace(0, max(noise_factors), 100)
        func = functions[chosen_extrapolator]
        popt, pcov = curve_fit(func, noise_factors, evs_noise_factors)
        y = func(x, *popt)
        axs_0[0].plot(x, y, alpha=0.2, color=blue)
        axs_1[row, col].plot(x, y, alpha=1.0)
    except:
        pass

    # Add extrapolated value at noise factor 0.
    noise_factors = [0.0] + noise_factors
    evs_noise_factors = [evs[qubit]] + list(evs_noise_factors)
    errors = [stds[qubit]] + list(stds_noise_factors)

    # Collated plot:
    axs_0[0].errorbar(
        noise_factors,
        evs_noise_factors,
        errors,
        marker="x",
        linestyle="",
        color=blue,
        alpha=0.2,
    )

    # Individual plots:
    axs_1[row, col].errorbar(
        noise_factors, evs_noise_factors, errors, fmt="x", alpha=1.0
    )
    axs_1[row, col].set_ylim(-1, 2)
    axs_1[row, col].axhline(1, color="k", linestyle="--")
    axs_1[row, col].set_title(f"Qubit: {qubit}\nExtrapolator: {chosen_extrapolator}")


# Raw and extrapolated values for each qubit.
axs_0[1].errorbar(
    range(100),
    evs,
    stds,
    marker="x",
    linestyle="",
    color=blue,
    label="Extrapolated",
)

axs_0[1].errorbar(
    range(100),
    result_dict["PubResult.data"]["evs_noise_factors"][:, 0],
    result_dict["PubResult.data"]["stds_noise_factors"][:, 0],
    marker="x",
    linestyle="",
    color=red,
    label="Noise factor: 1 (raw)",
)

axs_0[0].set_ylim(-1, 2)
axs_0[1].set_ylim(-1, 2)
axs_0[0].axhline(1, color="k", linestyle="--")
axs_0[1].axhline(1, color="k", linestyle="--")
axs_0[0].axhline(0, color="k", linestyle="-", linewidth=1)
axs_0[1].axhline(0, color="k", linestyle="-", linewidth=1)
axs_0[0].set_xlabel("Noise factor")
axs_0[0].set_ylabel(r"$\langle Z \rangle$")
axs_0[1].set_xlabel("Qubit")
axs_0[1].set_ylabel(r"$\langle Z \rangle_0$")
axs_0[1].legend()

for i in [0, 1]:
    for item in (
        [axs_0[i].xaxis.label, axs_0[i].yaxis.label]
        + axs_0[i].get_xticklabels()
        + axs_0[i].get_yticklabels()
    ):
        item.set_fontsize(12)

fig_0.tight_layout()
fig_1.tight_layout()

fn = os.path.join(plot_dir, "collated_ZNE_plot.pdf")
fig_0.savefig(fn, dpi=500)
print(f"Figure saved to: {fn}")
fn = os.path.join(plot_dir, "individual_ZNE_plots.pdf")
fig_1.savefig(fn, dpi=500)
print(f"Figure saved to: {fn}")
