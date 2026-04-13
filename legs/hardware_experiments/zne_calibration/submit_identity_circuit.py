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
import numpy as np
import os
from pathlib import Path
from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
import pickle as pkl
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_circuit_layout, plot_error_map
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    EstimatorV2,
    EstimatorOptions,
)

# Params
n = 100
layers = 6
extra_half_layer = True

# Account set-up
service = QiskitRuntimeService(
    instance="ADD INSTANCE CRN",
)

# Create directories
start_time = datetime.datetime.now()
start_time = str(start_time).replace(" ", "_").replace("-", "_").replace(":", "_")[:-7]
exp_dir = f"identity_circuit_experiments/{start_time}_n_{n}_layers_{layers if not extra_half_layer else layers + 0.5}/"
result_dir = os.path.join(exp_dir, "results")
fig_dir = os.path.join(exp_dir, "figures")
Path(exp_dir).mkdir(parents=True, exist_ok=True)
Path(result_dir).mkdir(parents=True, exist_ok=True)
Path(fig_dir).mkdir(parents=True, exist_ok=True)


# Two-qubit identity operator
I = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)


# Brickwork identity ansatz
qc = QuantumCircuit(n)

# Add two-qubit identities in a brickwall pattern.
for _ in range(layers):
    for i in range(0, n - 1, 2):
        qc.unitary(I, [i, i + 1])
    for i in range(1, n - 1, 2):
        qc.unitary(I, [i, i + 1])

if extra_half_layer:
    for i in range(0, n - 1, 2):
        qc.unitary(I, [i, i + 1])

ansatz, initial_parameters = generate_ansatz_from_circuit(
    qc, qubits_initially_zero=True
)

assigned_ansatz = ansatz.assign_parameters(initial_parameters)

# Observables: Z on all qubits.
observables = [SparsePauliOp("I" * (n - i - 1) + "Z" + "I" * i) for i in range(n)]


# Transpile
backend = service.backend("ibm_pittsburgh")

# We set optimization_level=1 so that the transpiler doesn't reduce the ansatz to a layer of single-
# qubit gates!
pass_manager = generate_preset_pass_manager(
    optimization_level=1, backend=backend, seed_transpiler=1234
)

isa_circuit = pass_manager.run(assigned_ansatz)
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

# # Zero-noise extrapolation with PEA (ZNE(PEA))
# options.experimental = {'resilience':{'zne': {'amplifier': 'pea'}}}
# options.resilience.layer_noise_learning.max_layers_to_learn = 4
# options.resilience.layer_noise_learning.num_randomizations = 32
# options.resilience.layer_noise_learning.shots_per_randomization = 128
# options.resilience.layer_noise_learning.layer_pair_depths = (0, 1, 2, 4, 16, 32)

estimator = EstimatorV2(mode=backend, options=options)

# Submit job
job = estimator.run([(isa_circuit, isa_observables)])

job_id = job.job_id()
print(f"Job ID: {job_id}")

result = job.result()
print(result)
print(f"Expectation values: {result[0].data.evs}")

# NOTE: I tried to pickle "result", however result[0].data is a DataBin class, which for
# some reason throws an error whilst un-pickling :(
result_dict = {
    "Job_ID": job_id,
    "PrimitiveResult.metadata": result.metadata,
    "PubResult.metadata": result[0].metadata,
    "PubResult.data": result[0].data.__dict__,
}

fn = os.path.join(
    result_dir, f"results_{backend.name}_{job_id}_identity_{layers}_layers.pkl"
)
with open(fn, "wb") as file:
    pkl.dump(result_dict, file)
