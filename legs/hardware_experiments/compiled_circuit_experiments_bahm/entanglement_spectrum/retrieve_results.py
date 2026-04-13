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
import numpy as np
import pickle as pkl

# Specify the job ID and the results directory it should have been saved to
result_dir = "./experiments/n_100_J0_1.0_J1_0.5_layers_3/2025_11_27_09_39_11/results"
job_id = "JOB ID"

service = QiskitRuntimeService(
    instance="ADD INSTANCE CRN",
)
job = service.job(job_id)
result = job.result()
print(result)
print(f"Expectation values: {result[0].data.evs}")

n = 100

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


# NOTE: I tried to pickle "result", however result[0].data is a DataBin class, which for
# some reason throws an error whilst un-pickling :(
result_dict = {
    "Job_ID": job_id,
    "observables": observables,
    "PrimitiveResult.metadata": result.metadata,
    "PubResult.metadata": result[0].metadata,
    "PubResult.data": result[0].data.__dict__,
}

fn = os.path.join(result_dir, f"tomography_results_{job.backend().name}_{job_id}.pkl")
with open(fn, "wb") as file:
    pkl.dump(result_dict, file)
