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


"""
Script to retrieve a result from the IBM quantum platform if the submission terminal closed before
the job completed.
"""

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.quantum_info import SparsePauliOp
import os
import pickle as pkl

# Specify the job ID and the results directory it should have been saved to
result_dir = "./experiments/n_100_J0_1.0_J1_0.5_layers_3/2025_11_27_09_35_54/results"

# NOTE: You must put the job IDs of the identity circuit experiment and the actual experiment
# separately. If the job ID corresponds to the identity circuit, set `identity_circuit = True`.
job_id = "JOB ID"

identity_circuit = False
if identity_circuit:
    name = "identity_circuit"
else:
    name = "compiled_circuit"

# NOTE: set `n`, `J0`, and `J1` to the correct values. Otherwise the Hamiltonian observable
# definition will be incorrect.
n = 100
J0 = 1.0
J1 = -2.0

service = QiskitRuntimeService(
    instance="ADD INSTANCE CRN",
)
job = service.job(job_id)
result = job.result()
print(result)
print(f"Expectation values: {result[0].data.evs}")

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


# NOTE: I tried to pickle "result", however result[0].data is a DataBin class, which for
# some reason throws an error whilst un-pickling :(
result_dict = {
    "Job_ID": job_id,
    "observables": identity_observables if identity_circuit else observables,
    "PrimitiveResult.metadata": result.metadata,
    "PubResult.metadata": result[0].metadata,
    "PubResult.data": result[0].data.__dict__,
}

fn = os.path.join(result_dir, f"results_{job.backend().name}_{job_id}_{name}.pkl")
with open(fn, "wb") as file:
    pkl.dump(result_dict, file)
