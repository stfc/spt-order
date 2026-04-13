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


import datetime
import os
from pathlib import Path
from qiskit.qasm3 import loads
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_circuit_layout, plot_error_map
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    EstimatorV2,
    EstimatorOptions,
)

# Account set-up
service = QiskitRuntimeService(
    instance="ADD INSTANCE CRN",
)

# Params
n = 100
layers = 6
J0 = 1.0
J1 = -2.0

# Compiled circuit fn
# "../../../bond_alternating_heisenberg_model_compiling/results/n_100_layers_3_J0_1.0_J1_0.5_tt_1e-12_1756375695282178000_fidelity_0.9900.txt"
# "../../../bond_alternating_heisenberg_model_compiling/results/n_100_layers_3_J0_1.0_J1_-1.0_tt_1e-12_1756388302044458855_fidelity_0.9900.txt"
# "../../../bond_alternating_heisenberg_model_compiling/results/n_100_layers_6_J0_1.0_J1_-2.0_tt_1e-12_1756388702662351325_fidelity_0.9794.txt"

ground_state_file = "../../../bond_alternating_heisenberg_model_compiling/results/n_100_layers_6_J0_1.0_J1_-2.0_tt_1e-12_1756388702662351325_fidelity_0.9794.txt"

assert f"n_{n}" in ground_state_file
assert f"layers_{layers}" in ground_state_file
assert f"J0_{J0}" in ground_state_file
assert f"J1_{J1}" in ground_state_file

with open(ground_state_file, "r") as file:
    qasm_string = file.read()

qc = loads(qasm_string)


# Create directories
start_time = datetime.datetime.now()
start_time = str(start_time).replace(" ", "_").replace("-", "_").replace(":", "_")[:-7]
exp_dir = f"experiments/n_{n}_J0_{J0}_J1_{J1}_layers_{layers}/{start_time}/"
result_dir = os.path.join(exp_dir, "results")
fig_dir = os.path.join(exp_dir, "figures")
Path(exp_dir).mkdir(parents=True, exist_ok=True)
Path(result_dir).mkdir(parents=True, exist_ok=True)
Path(fig_dir).mkdir(parents=True, exist_ok=True)


# Observables: all Pauli strings.
pauli_ops = ["I", "X", "Y", "Z"]

# Maximum segment length.
l = 4

# Maximum number of segments (tile length-l segments to fill the whole chain).
num_segments = n // l
observables = [0] * 4**l * num_segments

# Construct all l-qubit Pauli strings for each l-qubit segment of the chain.
for segment in range(num_segments):
    for k in range(4**l):
        if k == 0:
            indices = "0" * l
        else:
            # Count in base-4.
            pad = int(l - 1 - np.floor(np.log(k) / np.log(4)))
            indices = str(np.base_repr(k, 4, pad))[::-1]

        # Pauli string, big-endian.
        pauli_string = "".join([pauli_ops[int(i)] for i in indices])

        observables[segment * 4**l + k] = SparsePauliOp.from_sparse_list(
            [(pauli_string, range(segment * l, segment * l + l), 1)], n
        )


# Check number of commuting groups.
sum_obs = SparsePauliOp.sum(observables)
print(f"{len(observables)} observables in total.")
print(f"{len(sum_obs.group_commuting())} commuting groups.")

# Transpile
backend = service.backend("ibm_pittsburgh")

pass_manager = generate_preset_pass_manager(
    optimization_level=3, backend=backend, seed_transpiler=1234
)

isa_circuit = pass_manager.run(qc)
layout = isa_circuit.layout
isa_circuit.draw("mpl", idle_wires=False, fold=-1)
plt.savefig(os.path.join(fig_dir, "transpiled_circuit.pdf"), dpi=100)
print(
    f"Transpiled CX depth: {isa_circuit.depth(filter_function=lambda instr: len(instr.qubits) > 1)}"
)
print(f"Transpiled ops: {isa_circuit.count_ops()}")

plot_error_map(backend)
plt.savefig(os.path.join(fig_dir, "error_map.pdf"), dpi=100)
plot_circuit_layout(isa_circuit, backend, view="virtual")
plt.savefig(os.path.join(fig_dir, "layout_virtual.pdf"), dpi=100)
plot_circuit_layout(isa_circuit, backend, view="physical")
plt.savefig(os.path.join(fig_dir, "layout_physical.pdf"), dpi=100)

# Apply circuit layout to observables
isa_observables = [observable.apply_layout(layout) for observable in observables]


# Execution
options = EstimatorOptions(default_shots=10000, resilience_level=0)

# Readout error mitigation (TREX)
options.resilience.measure_mitigation = True

# Pauli twirling (PT)
options.twirling.enable_gates = True
options.twirling.num_randomizations = 100

# Zero-noise extrapolation (ZNE)
options.resilience.zne_mitigation = True
options.resilience.zne.noise_factors = (1, 1.05, 1.1, 1.15, 1.2, 1.4, 1.6, 1.8, 2.0)
options.resilience.zne.extrapolator = ("exponential", "polynomial_degree_2", "linear")

estimator = EstimatorV2(mode=backend, options=options)

# Submit job
job = estimator.run([(isa_circuit, isa_observables)])

job_id = job.job_id()
print(f"Job ID: {job_id}")

result = job.result()
job_id = job.job_id()
print(result)
print(f"Expectation values: {result[0].data.evs}")

# NOTE: I tried to pickle "result", however result[0].data is a DataBin class, which for
# some reason throws an error whilst un-pickling :(
result_dict = {
    "Job_ID": job_id,
    "observables": observables,
    "PrimitiveResult.metadata": result.metadata,
    "PubResult.metadata": result[0].metadata,
    "PubResult.data": result[0].data.__dict__,
}

fn = os.path.join(result_dir, f"tomography_results_{backend.name}_{job_id}.pkl")
with open(fn, "wb") as file:
    pkl.dump(result_dict, file)
