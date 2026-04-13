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
from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_circuit_layout, plot_error_map
from qiskit_ibm_runtime import (
    Batch,
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
layers = 3
extra_half_layer = False
J0 = 0.5
J1 = 1.0

# Compiled circuit fn
# "../../bond_alternating_heisenberg_model_compiling/results/n_100_layers_3_J0_1.0_J1_0.5_tt_1e-12_1756375695282178000_fidelity_0.9900.txt"
# "../../bond_alternating_heisenberg_model_compiling/results/n_100_layers_3_J0_1.0_J1_-1.0_tt_1e-12_1756388302044458855_fidelity_0.9900.txt"
# "../../bond_alternating_heisenberg_model_compiling/results/n_100_layers_6_J0_1.0_J1_-2.0_tt_1e-12_1756388702662351325_fidelity_0.9794.txt"
# "../../bond_alternating_heisenberg_model_compiling/results/n_100_layers_3_J0_0.5_J1_1.0_tt_1e-12_1769679224814412000_fidelity_0.9900.txt"

ground_state_file = "../../bond_alternating_heisenberg_model_compiling/results/n_100_layers_3_J0_0.5_J1_1.0_tt_1e-12_1769679224814412000_fidelity_0.9900.txt"

assert f"n_{n}" in ground_state_file
assert f"layers_{layers}" in ground_state_file
assert f"J0_{J0}" in ground_state_file
assert f"J1_{J1}" in ground_state_file

with open(ground_state_file, "r") as file:
    qasm_string = file.read()

qc = loads(qasm_string)

# Construct identity circuit for validation.

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
identity_qc = QuantumCircuit(n)

# Add two-qubit identities in a brickwall pattern.
for _ in range(layers):
    for i in range(0, n - 1, 2):
        identity_qc.unitary(I, [i, i + 1])
    for i in range(1, n - 1, 2):
        identity_qc.unitary(I, [i, i + 1])

if extra_half_layer:
    for i in range(0, n - 1, 2):
        identity_qc.unitary(I, [i, i + 1])

identity_ansatz, identity_parameters = generate_ansatz_from_circuit(
    identity_qc, qubits_initially_zero=True
)

identity_qc = identity_ansatz.assign_parameters(identity_parameters)

# Create directories
start_time = datetime.datetime.now()
start_time = str(start_time).replace(" ", "_").replace("-", "_").replace(":", "_")[:-7]
exp_dir = f"experiments/n_{n}_J0_{J0}_J1_{J1}_layers_{layers}/{start_time}/"
result_dir = os.path.join(exp_dir, "results")
fig_dir = os.path.join(exp_dir, "figures")
Path(exp_dir).mkdir(parents=True, exist_ok=True)
Path(result_dir).mkdir(parents=True, exist_ok=True)
Path(fig_dir).mkdir(parents=True, exist_ok=True)


# Observables:
# Zi on all qubits.
z_observables = [SparsePauliOp.from_sparse_list([("Z", [i], 1)], n) for i in range(n)]

# ZiZj with i=0,...,99 and j=i+1,...,i+20 (obviously not going beyond 99).
zz_observables = []
for i in range(n):
    max_index = min(i + 20, n - 1)
    for j in range(i + 1, max_index + 1):
        zz_observables.append(SparsePauliOp.from_sparse_list([("ZZ", [i, j], 1)], n))

# BAHM Hamiltonian: J_{i%2}(XX + YY + ZZ)_{i, i+1}
hamiltonian_term_list = []
for i in range(n - 1):
    for term in ["XX", "YY", "ZZ"]:
        hamiltonian_term_list.append(
            (term, [i, i + 1], J0 / 4 if i % 2 == 0 else J1 / 4)
        )
hamiltonian_observable = [SparsePauliOp.from_sparse_list(hamiltonian_term_list, n)]

# Define multiple s values to compute the string order starting at multiple sites.
s_list = [20, 30, 40, 50, 60]
num_obs = 10
even_string_order_observables = [
    SparsePauliOp.from_sparse_list(
        [("Z" * 2 * (j + 1), list(range(s, s + 2 * (j + 1))), (-1) ** (j + 1))], n
    )
    for s in s_list
    for j in range(num_obs)
]
odd_string_order_observables = [
    SparsePauliOp.from_sparse_list(
        [("Z" * 2 * (j + 1), list(range(s + 1, s + 2 * (j + 1) + 1)), (-1) ** (j + 1))],
        n,
    )
    for s in s_list
    for j in range(num_obs)
]

observables = (
    z_observables
    + zz_observables
    + hamiltonian_observable
    + even_string_order_observables
    + odd_string_order_observables
)

identity_observables = z_observables

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
identity_isa_observables = [
    observable.apply_layout(layout) for observable in identity_observables
]

# Apply circuit layout to identity circuit
pass_manager = generate_preset_pass_manager(
    optimization_level=1,
    backend=backend,
    seed_transpiler=1234,
    initial_layout=layout.final_index_layout(),
)
identity_isa_circuit = pass_manager.run(identity_qc)

identity_isa_circuit.draw("mpl", idle_wires=False, fold=-1)
plt.savefig(os.path.join(fig_dir, "transpiled_identity_circuit.pdf"), dpi=100)
print(
    f"Identity transpiled CX depth: {identity_isa_circuit.depth(filter_function=lambda instr: len(instr.qubits) > 1)}"
)
print(f"Identity transpiled ops: {identity_isa_circuit.count_ops()}")

# Check that both circuits use the same layout.
assert identity_isa_circuit.layout == isa_circuit.layout

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


with Batch(backend=backend) as batch:
    estimator = EstimatorV2(mode=batch, options=options)
    jobs = {}

    for name, circuit in zip(
        ["identity_circuit", "compiled_circuit"], [identity_isa_circuit, isa_circuit]
    ):

        # Submit job
        job = estimator.run(
            [
                (
                    circuit,
                    (
                        identity_isa_observables
                        if name == "identity_circuit"
                        else isa_observables
                    ),
                )
            ]
        )

        job_id = job.job_id()
        print(f"Job ID: {job_id}, circuit: {name}")

        jobs[name] = job


for name, job in jobs.items():
    result = job.result()
    job_id = job.job_id()
    print(job_id)
    print(name)
    print(result)
    print(f"Expectation values: {result[0].data.evs}")

    # NOTE: I tried to pickle "result", however result[0].data is a DataBin class, which for
    # some reason throws an error whilst un-pickling :(
    result_dict = {
        "Job_ID": job_id,
        "observables": (
            identity_observables if name == "identity_circuit" else observables
        ),
        "PrimitiveResult.metadata": result.metadata,
        "PubResult.metadata": result[0].metadata,
        "PubResult.data": result[0].data.__dict__,
    }

    fn = os.path.join(result_dir, f"results_{backend.name}_{job_id}_{name}.pkl")
    with open(fn, "wb") as file:
        pkl.dump(result_dict, file)
