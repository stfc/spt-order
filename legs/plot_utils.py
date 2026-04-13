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

import time
from pathlib import Path

import matplotlib.pyplot as plt


# Returns a list of Boolean values
# The ith element will be True if all the values in the ith dataframe column are equal
def single_val_cols(df):
    a = df.drop(["fidelity_history"], axis=1)
    a = a.to_numpy()
    return (a[0] == a).all(0)


# Splits string into n items per line, items separated by commas
def make_n_items_per_line(input_str, n):
    lst = input_str.split(",")
    split_str = ""
    for pos, item in enumerate(lst[:-1]):
        split_str += item
        if (pos + 1) % n == 0:
            split_str += ",\n"
        else:
            split_str += ", "
    split_str = split_str[:-2]
    return split_str


def get_labels(variable_data, runs):
    labels = []
    for i in range(runs):
        label = ""
        for j in range(len(variable_data.columns)):
            label += f"{variable_data[variable_data.keys()[j]][i]},"
        label = make_n_items_per_line(label, 4)
        labels.append(label)

    return labels


def plot_and_save(data, max_runs, plot_directory=None):
    num_qubits_list = []
    for num in data["qubits"]:
        if num not in num_qubits_list:
            num_qubits_list.append(num)

    for num_qubits in num_qubits_list:
        filtered_data = (
            data.loc[data["qubits"] == num_qubits]
            .iloc[-max_runs:]
            .sort_values(by=["layers"])
            .reset_index(drop=True)
        )

        single_val_cols_list = single_val_cols(filtered_data)
        single_val_keys_list = []
        title_string = ""
        top_label = ""

        # For each column in the data frame, if all the values in this column are equal, append the column
        # name and corresponding value to a string. If the values in the column are not all equal, store the
        # column key.
        for i, bool in enumerate(single_val_cols_list):
            if bool:
                title_string += f"{filtered_data.keys()[i]}={filtered_data[filtered_data.keys()[i]][0]},"
            else:
                single_val_keys_list.append(filtered_data.keys()[i])
                top_label += f"{filtered_data.keys()[i]}, "

        # Split strings into multiple lines
        title_string = make_n_items_per_line(title_string, 5)
        top_label = make_n_items_per_line(top_label, 4)

        # Get all non-single-value-column data
        variable_data = filtered_data[single_val_keys_list]

        # Plot the global cost history for each recompilation, as a function of 1. layer count and 2. CNOT depth
        # Parameters common to all runs go in the title
        # Parameters not common to all runs go in the legend
        iter_count = 0
        plt.rcParams["figure.figsize"] = 10, 8
        fig1, ax1 = plt.subplots()
        runs = len(filtered_data["fidelity_history"])
        labels = get_labels(variable_data, runs)
        for i in range(runs):
            fidelity = filtered_data["fidelity_history"][i]
            ax1.plot([1 - f for f in fidelity], label=labels[i])
            if len(fidelity) > iter_count:
                iter_count = len(fidelity)

        ax1.set_title(title_string, fontsize=8)
        ax1.hlines(1e-2, 0, iter_count - 1, linestyles="dashed", colors="black")
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(  # Remove duplicate labels if in legend
            by_label.values(),
            by_label.keys(),
            loc="upper right",
            title=top_label,
            framealpha=0.3,
        )
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("1 - Fidelity")
        ax1.set_yscale("log")
        ax1.set_xscale("log")
        ax1.grid(True, "both")

        if plot_directory is None:
            plot_directory = "./plots/"
        Path(plot_directory).mkdir(parents=True, exist_ok=True)

        fig1_name = (
            f"{str(time.time_ns())[:-9]}_{num_qubits}_qubits_fidelity_vs_iteration.png"
        )
        fig1.savefig(plot_directory + fig1_name, bbox_inches="tight", dpi=500)
        print("Saved figure to " + fig1_name)
