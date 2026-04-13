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
import re
import sys

import pandas as pd
import pickle as pkl

from legs.plot_utils import plot_and_save


def extract_fidelity_history_and_settings(log_file):
    settings_pattern = re.compile(r"Namespace\((.*?)\)")

    initial_fidelity_pattern = re.compile(r"Initial fidelity: (\d+\.?\d*)")
    fidelity_pattern = re.compile(r"Intermediate result: Fidelity (\d+\.?\d*)")

    fidelity_values = []
    settings = None
    with open(log_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            settings_match = settings_pattern.search(line)
            if settings_match and settings is None:
                settings = settings_match.group(1)
            initial_fidelity_match = initial_fidelity_pattern.search(line)
            if initial_fidelity_match:
                fidelity_values.append(float(initial_fidelity_match.group(1)))
            fidelity_match = fidelity_pattern.search(line)
            if fidelity_match:
                fidelity_values.append(float(fidelity_match.group(1)))

    return settings, fidelity_values


def scrape_all_logs(logs_directory):
    logs = [
        log
        for log in os.listdir(logs_directory)
        if log.startswith("log") and log.endswith(".log")
    ]
    experiment_results = []
    for log_file in logs:
        settings, fidelity_history = extract_fidelity_history_and_settings(
            os.path.join(logs_directory, log_file)
        )

        experiment_results.append((settings_to_dict(settings), fidelity_history))
    sorted_results = sorted(experiment_results, key=lambda x: x[-1][-1])
    return sorted_results


def read_checkpoints(directory):
    runs = os.listdir(directory)
    experiment_results = []
    for run in runs:
        settings = None
        checkpoints = [
            f
            for f in sorted(
                os.listdir(os.path.join(directory, run)),
                key=lambda x: int(re.findall(r"\d+", x)[0]),
            )
        ]
        fidelity_history = []
        for chkpt in checkpoints:
            print(chkpt)
            with open(os.path.join(directory, run, chkpt), "rb") as file:
                data = pkl.load(file)
            if settings is None:
                settings = data["args"]
            fidelity_history.append(1 - data["intermediate_result"].fun)
        experiment_results.append((settings, fidelity_history))
    sorted_results = sorted(experiment_results, key=lambda x: x[-1][-1])
    return sorted_results


def settings_to_dict(settings_str):
    # Split by commas to get individual key-value pairs
    pairs = settings_str.split(", ")
    # Initialize an empty dictionary to hold the settings
    settings_dict = {}
    # Process each pair
    for pair in pairs:
        key, value = pair.split("=")
        # Convert value to the appropriate type
        if value.isdigit():  # Check if value is an integer
            value = int(value)
        else:
            try:
                value = float(value)  # Check if value is a float
            except ValueError:
                if value.lower() == "true":  # Handle boolean values
                    value = True
                elif value.lower() == "false":
                    value = False
                elif value.startswith("'") and value.endswith(
                    "'"
                ):  # Handle string values enclosed in single quotes
                    value = value.strip("'")
                else:
                    value = value  # Keep as string for anything else
        # Add to the dictionary
        settings_dict[key] = value
    return settings_dict


def scrape_and_plot(directory, logs_or_checkpoints, max_runs=10):
    data = pd.DataFrame()
    if logs_or_checkpoints == "logs":
        sorted_results = scrape_all_logs(directory)
    elif logs_or_checkpoints == "checkpoints":
        sorted_results = read_checkpoints(directory)
    for settings, fidelity_history in sorted_results:
        settings["fidelity_history"] = fidelity_history
        dict_df = pd.DataFrame([settings])
        data = pd.concat([data, dict_df], ignore_index=True)

    # Plotting depends on these being the final two columns
    data.insert(len(data.columns) - 1, "fidelity_history", data.pop("fidelity_history"))

    plot_and_save(data, max_runs)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise ValueError(
            "Usage: python plot_cost_histories_from_logs_or_checkpoints.py <max_runs_to_plot> <directory> <logs or checkpoints>"
        )
    else:
        max_runs = int(sys.argv[1])
        directory = str(sys.argv[2])
        logs_or_checkpoints = str(sys.argv[3])

    # NOTE: checkpoint plotting only works for new checkpoints including the args_dict.
    # NOTE: currently plotting from checkpoints does not include the initial cost.
    scrape_and_plot(directory, logs_or_checkpoints, max_runs)
