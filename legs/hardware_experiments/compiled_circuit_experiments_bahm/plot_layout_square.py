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
import os
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.visualization import plot_coupling_map
from qiskit_ibm_runtime import QiskitRuntimeService
import matplotlib.pyplot as plt
from legs.hardware_experiments.compiled_circuit_experiments_bahm.constants import (
    EXPERIMENT_TO_PLOT,
)

blue = "#00A2FF"
black = "#808080"


def get_pittsburgh_qubit_coordinates():
    coords = []

    num_rows = 15
    for row in range(num_rows):
        if row % 2 == 0:
            # Alternating rows of 16 qubits
            for col in range(16):
                coords.append([num_rows - row, col])
        elif row % 4 == 1:
            # Row of 4 qubits, offset by 3
            for col in range(4):
                coords.append([num_rows - row, 4 * col + 3])
        else:
            # Row of 4 qubits, offset by 1
            for col in range(4):
                coords.append([num_rows - row, 4 * col + 1])

    return coords


# Account set-up
service = QiskitRuntimeService(
    instance="ADD INSTANCE CRN",
)

# File paths
# Add the result and target MPS file paths to filepaths.json.
with open("filepaths.json", mode="r") as file:
    experiments = json.load(file)

result_fn = experiments[str(EXPERIMENT_TO_PLOT)]["result_fn"]

# Save figures in the right directory
split_fn = result_fn.split("/")
results_index = split_fn.index("results")
plot_dir = "/".join(split_fn[:results_index]) + "/figures/"
Path(plot_dir).mkdir(parents=True, exist_ok=True)

# Retrieve job ID
split_fn = result_fn.split("_")
job_id_index = split_fn.index("compiled") - 1
job_id = split_fn[job_id_index]


# Retrieve circuit from job
job = service.job(job_id)
qc: QuantumCircuit = job.inputs["pubs"][0][0]

# Dict of physical qubit index vs virtual qubit objects (including ancillae)
layout = qc.layout.initial_layout._p2v

# Dict of {"virtual qubit index": "physical qubit index"}, not including ancillae
virtual_to_physical = {}
for physical_qubit, virtual_qubit in layout.items():
    if virtual_qubit._register.name == "q":
        virtual_to_physical[virtual_qubit._index] = physical_qubit
    else:
        pass


pittsburgh_qubit_coordinates = get_pittsburgh_qubit_coordinates()

backend = service.backend("ibm_pittsburgh")

coupling_map = backend.coupling_map.get_edges()

# Make all virtual qubits black with virtual index label, rest blue without label
qubit_colours = [black] * 156
qubit_labels = [""] * 156
for index, virtual_qubit in layout.items():
    if virtual_qubit._register.name == "q":
        qubit_colours[index] = blue
        qubit_labels[index] = virtual_qubit._index

# Make edges between virtual qubits i, i+1 black.
line_color = [black] * len(coupling_map)
for i, [q0, q1] in enumerate(coupling_map):
    virtual_q0 = layout[q0]
    virtual_q1 = layout[q1]
    if virtual_q0._register.name == "q" and virtual_q1._register.name == "q":
        # Both are qubits in the chain
        if int(abs(virtual_q0._index - virtual_q1._index)) == 1:
            line_color[i] = blue

plot_coupling_map(
    156,
    qubit_coordinates=pittsburgh_qubit_coordinates,
    coupling_map=coupling_map,
    qubit_color=qubit_colours,
    line_color=line_color,
    qubit_labels=qubit_labels,
    qubit_size=50,
    line_width=10,
    font_size=22,
    font_color="black",
)
plt.savefig(os.path.join(plot_dir, "layout_virtual_square.pdf"), dpi=100)
