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


import logging
import pickle
import os
import numpy as np
import quimb.tensor as qtn
from functools import partial
from pathlib import Path
from argparse import ArgumentParser
from legs.optimizers import adam
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit
from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
from qiskit_addon_aqc_tensor.simulation.aer import QiskitAerMPS
from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator
from qiskit_aer import AerSimulator
from scipy.optimize import OptimizeResult, minimize
from adaptaqc.utils.utilityfunctions import qiskit_to_tenpy_mps
from qiskit_addon_aqc_tensor.simulation import (
    compute_overlap,
    tensornetwork_from_circuit,
)

parser = ArgumentParser()
# --------------------------------------------------------------------------------
# Misc

parser.add_argument(
    "-cd",
    "--checkpoint_dir",
    type=str,
    default="./checkpoint/",
    help="Specific checkpoint directory.",
)

parser.add_argument(
    "-dpc",
    "--delete_prev_chkpt",
    type=str,
    default="False",
    help="Whether to delete the previous checkpoint after saving a new one.",
)

parser.add_argument(
    "-ld",
    "--log_dir",
    type=str,
    default="./logs/",
    help="Top-level directory for logs.",
)


args = parser.parse_args()


# Reconstruct timestamp when job was created
TIME = args.checkpoint_dir.split("_")[-1].replace("/", "")


# Load checkpoint
checkpoint_names = os.listdir(args.checkpoint_dir)
int_checkpoints = [int(cn[6:-4]) for cn in checkpoint_names]  # assumes chkpt_{int}.pkl
loaded_iter_number = max(int_checkpoints)
checkpoint_fn = args.checkpoint_dir + f"chkpt_{loaded_iter_number}.pkl"

with open(checkpoint_fn, "rb") as chkpt_f:
    checkpoint = pickle.load(chkpt_f)

original_args = checkpoint["args"]
intermediate_result = checkpoint["intermediate_result"]


# --------------------------------------------------------------------------------
# Helper funcations
def get_backend():
    if original_args["backend"] == "MPS_SIM":
        return AerSimulator(
            method="matrix_product_state",
            matrix_product_state_truncation_threshold=original_args["mps_truncation"],
            matrix_product_state_max_bond_dimension=original_args[
                "mps_max_bond_dimension"
            ],
        )
    elif original_args["backend"] == "QUIMB":
        return QuimbSimulator(
            partial(
                qtn.CircuitMPS,
                gate_opts={
                    "cutoff": original_args["mps_truncation"],
                    "max_bond": original_args["mps_max_bond_dimension"],
                },
            ),
            autodiff_backend="jax",
        )
    else:
        raise ValueError("Unrecognized backend")


def get_optimisation_method():
    if original_args["optimiser_method"] == "adam":
        return adam
    elif original_args["optimiser_method"] == "lbfgsb":
        return "L-BFGS-B"


def str_to_bool(arg: str):
    return arg.strip().lower() == "true"


n = original_args["qubits"]
J0 = original_args["even_coupling"]
J1 = original_args["odd_coupling"]
target_trunc = original_args["target_trunc"]
layers = original_args["layers"]
filename_parts = [
    f"n_{n}",
    f"layers_{layers}",
    f"J0_{J0}",
    f"J1_{J1}",
    f"tt_{target_trunc}",
    f"{TIME}",
]
filename = "_".join(filename_parts)

# Add to original log file
log_file = os.path.join(original_args["log_dir"], f"log-{filename}.log")
logging.basicConfig(filename=log_file, filemode="a")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(f"Loaded: {checkpoint_fn}")
logger.info(args)


# Reconstruct the ansatz
target_dir = "saved_mps/"
target_fn = f"target_n_{n}_J0_{J0}_J1_{J1}_hz_0.0_trunc_{target_trunc}_compressed_bd_{original_args["compressed_bond_dim"] if original_args["compressed_bond_dim"] > 0 else None}.pkl"

with open(target_dir + target_fn, "rb") as mps_f:
    raw_target_mps = pickle.load(mps_f)

logger.info(
    f"Loaded MPS of max bond dimension: {max([len(a) for a in raw_target_mps[1]])}"
)

qc = QuantumCircuit(n)
for _ in range(layers):
    for i in range(0, n - 1, 2):
        qc.cx(i, i + 1)
    for i in range(1, n - 1, 2):
        qc.cx(i, i + 1)

# Initialise as a product of Bell pairs: |01> - |10>
for i in range(0, n - 1, 2):
    qc.x([i, i + 1])
    qc.h(i)
    qc.cx(i, i + 1)

aqc_ansatz, _ = generate_ansatz_from_circuit(qc, qubits_initially_zero=True)

# Initialise with the checkpoint params
intermediate_params = intermediate_result.x

logger.info(
    f"Ansatz created with CX depth:{(aqc_ansatz.decompose(reps=3).depth(lambda gate: len(gate.qubits) > 1))}"
)

intermediate_fidelity = (
    abs(
        compute_overlap(
            tensornetwork_from_circuit(
                transpile(
                    aqc_ansatz.assign_parameters(intermediate_params),
                    basis_gates=["cx", "rx", "ry", "rz"],
                ),
                AerSimulator(
                    method="matrix_product_state",
                    matrix_product_state_truncation_threshold=1e-16,
                ),
            ),
            QiskitAerMPS(raw_target_mps[0], raw_target_mps[1]),
        )
    )
    ** 2
)

logger.info(f"Intermediate fidelity: {intermediate_fidelity}")

backend = get_backend()

# Get the target MPS in the correct format.
if isinstance(backend, AerSimulator):
    target_mps: QiskitAerMPS = QiskitAerMPS(raw_target_mps[0], raw_target_mps[1])
elif isinstance(backend, QuimbSimulator):
    # Raw MPS to Tenpy
    tenpy_mps = qiskit_to_tenpy_mps(raw_target_mps, return_form="SpinHalfSite")
    # Convert Tenpy MPS to Quimb
    arrays = [
        tenpy_mps.get_B(i, form="A").itranspose(["vL", "vR", "p"]).to_ndarray()
        for i in range(n)
    ]

    arrays[0] = np.squeeze(arrays[0], axis=0)
    arrays[-1] = np.squeeze(arrays[-1], axis=1)

    quimb_mps = qtn.tensor_1d.MatrixProductState(arrays=arrays, shape="lrp")
    quimb_mps.left_canonize()

    target_mps = quimb_mps


objective = MaximizeStateFidelity(target_mps, aqc_ansatz, backend)


def callback(intermediate_result: OptimizeResult):
    logger.info(f"Intermediate result: Fidelity {1 - intermediate_result.fun}")

    checkpoint_names = os.listdir(args.checkpoint_dir)
    int_checkpoints = [
        int(cn[6:-4]) for cn in checkpoint_names
    ]  # assumes chkpt_{int}.pkl
    if int_checkpoints:
        iter_number = max(int_checkpoints) + 1
    else:
        iter_number = 0

    checkpoint_data = {
        "args": original_args,
        "intermediate_result": intermediate_result,
    }

    with open(args.checkpoint_dir + f"chkpt_{iter_number}.pkl", "wb") as chkpt_f:
        pickle.dump(checkpoint_data, chkpt_f)

    if str_to_bool(args.delete_prev_chkpt) and iter_number > 0:
        os.remove(args.checkpoint_dir + f"chkpt_{iter_number - 1}.pkl")

    if intermediate_result.fun < original_args["sufficient_cost"]:
        # Good enough for now
        raise StopIteration


result = minimize(
    objective.loss_function,
    intermediate_params,
    method=get_optimisation_method(),
    jac=True,
    options={
        "maxiter": original_args["max_iter"] - (loaded_iter_number + 1),
        "ftol": np.nan,
        "gtol": np.sqrt(np.finfo(float).tiny),
    },
    callback=callback,
)

logger.info(result)

result_dir = "results/"
Path(result_dir).mkdir(parents=True, exist_ok=True)
result_data = {
    "args": original_args,
    "result": result,
    "target_mps": target_mps,
    "aqc_ansatz": aqc_ansatz,
}

with open(os.path.join(result_dir, filename) + ".pkl", mode="wb") as file:
    pickle.dump(result_data, file)
