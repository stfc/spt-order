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
import time
import numpy as np
import os
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

TIME = str(time.time_ns())

parser = ArgumentParser()
# --------------------------------------------------------------------------------
# Circuit / Hamiltonian parameters
parser.add_argument(
    "-n", "--qubits", type=int, default=100, help="Number of qubits (default 100)"
)

parser.add_argument(
    "-J0",
    "--even_coupling",
    type=float,
    default=1.0,
    help="Hamiltonian even-odd coupling (default 1.0)",
)
parser.add_argument(
    "-J1",
    "--odd_coupling",
    type=float,
    default=1.0,
    help="Hamiltonian odd-even coupling (default 1.0)",
)

parser.add_argument(
    "-tt",
    "--target_trunc",
    type=float,
    default=1e-8,
    help="MPS truncation threshold used to generate the target",
)

parser.add_argument(
    "-cbd",
    "--compressed_bond_dim",
    type=int,
    default=0,
    help="Bond dimension that the target MPS has been compressed to.",
)

# --------------------------------------------------------------------------------
# Compiling options
parser.add_argument(
    "-l",
    "--layers",
    type=int,
    default=5,
    help="Number of brickwall ansatz layers to add",
)
parser.add_argument(
    "-mi",
    "--max_iter",
    type=int,
    default=100_000,
    help="Maximum number of optimisation iterations to take",
)
parser.add_argument(
    "-sc",
    "--sufficient_cost",
    type=float,
    default=1e-2,
    help="Cost function value that AQCTensor will terminate (default 1e-2)",
)
parser.add_argument(
    "-b",
    "--backend",
    type=str,
    choices=["MPS_SIM", "QUIMB"],
    default="MPS_SIM",
)
parser.add_argument(
    "-om",
    "--optimiser_method",
    type=str,
    choices=["lbfgsb", "adam"],
    default="lbfgsb",
)

# --------------------------------------------------------------------------------
# Misc

parser.add_argument(
    "-mpst",
    "--mps_truncation",
    type=float,
    default=1e-8,
    help="Truncation to be used by Aer MPS simulator (default 1e-8)",
)

parser.add_argument(
    "-mpsbd",
    "--mps_max_bond_dimension",
    type=float,
    default=100,
    help="Max bond dimension to be used by Aer MPS simulator (default 100)",
)

parser.add_argument(
    "-cd",
    "--checkpoint_dir",
    type=str,
    default="./checkpoint/",
    help="Directory for checkpoints.",
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
    help="Directory for logs.",
)


# --------------------------------------------------------------------------------
# Helper funcations
def get_backend():
    if args.backend == "MPS_SIM":
        return AerSimulator(
            method="matrix_product_state",
            matrix_product_state_truncation_threshold=args.mps_truncation,
            matrix_product_state_max_bond_dimension=args.mps_max_bond_dimension,
        )
    elif args.backend == "QUIMB":
        return QuimbSimulator(
            partial(
                qtn.CircuitMPS,
                gate_opts={
                    "cutoff": args.mps_truncation,
                    "max_bond": args.mps_max_bond_dimension,
                },
            ),
            autodiff_backend="jax",
        )
    else:
        raise ValueError("Unrecognized backend")


def get_optimisation_method():
    if args.optimiser_method == "adam":
        return adam
    elif args.optimiser_method == "lbfgsb":
        return "L-BFGS-B"


def str_to_bool(arg: str):
    return arg.strip().lower() == "true"


args = parser.parse_args()
args_dict = vars(args)
n = args.qubits
J0 = args.even_coupling
J1 = args.odd_coupling
target_trunc = args.target_trunc
filename_parts = [
    f"n_{n}",
    f"layers_{args.layers}",
    f"J0_{J0}",
    f"J1_{J1}",
    f"tt_{target_trunc}",
    f"{TIME}",
]
filename = "_".join(filename_parts)

# Redirect logs with descriptive filename
Path(args.log_dir).mkdir(parents=True, exist_ok=True)
log_file = os.path.join(args.log_dir, f"log-{filename}.log")
logging.basicConfig(filename=log_file, filemode="a")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(filename)
logger.info(args)

target_dir = "saved_mps/"
target_fn = f"target_n_{n}_J0_{J0}_J1_{J1}_hz_0.0_trunc_{target_trunc}_compressed_bd_{args.compressed_bond_dim if args.compressed_bond_dim > 0 else None}.pkl"

with open(target_dir + target_fn, "rb") as mps_f:
    raw_target_mps = pickle.load(mps_f)

logger.info(
    f"Loaded MPS of max bond dimension: {max([len(a) for a in raw_target_mps[1]])}"
)

qc = QuantumCircuit(n)
for _ in range(args.layers):
    for i in range(0, n - 1, 2):
        qc.cx(i, i + 1)
    for i in range(1, n - 1, 2):
        qc.cx(i, i + 1)

# Initialise as a product of Bell pairs: |01> - |10>. Change 0 to 1 for odd-Haldane phase.
for i in range(0, n - 1, 2):
    qc.x([i, i + 1])
    qc.h(i)
    qc.cx(i, i + 1)


aqc_ansatz, aqc_initial_parameters = generate_ansatz_from_circuit(
    qc, qubits_initially_zero=True
)

logger.info(
    f"Ansatz created with CX depth:{(aqc_ansatz.decompose(reps=3).depth(lambda gate: len(gate.qubits) > 1))}"
)

initial_fidelity = (
    abs(
        compute_overlap(
            tensornetwork_from_circuit(
                transpile(
                    aqc_ansatz.assign_parameters(aqc_initial_parameters),
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

logger.info(f"Initial fidelity: {initial_fidelity}")

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


checkpoint_dir = os.path.join(args.checkpoint_dir, f"{filename}/")
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)


def callback(intermediate_result: OptimizeResult):
    logger.info(f"Intermediate result: Fidelity {1 - intermediate_result.fun}")

    try:
        iter_number = intermediate_result.nit
    except AttributeError:
        checkpoint_names = os.listdir(checkpoint_dir)
        int_checkpoints = [
            int(cn[6:-4]) for cn in checkpoint_names
        ]  # assumes chkpt_{int}.pkl
        if int_checkpoints:
            iter_number = max(int_checkpoints) + 1
        else:
            iter_number = 0

    checkpoint_data = {"args": args_dict, "intermediate_result": intermediate_result}

    with open(checkpoint_dir + f"chkpt_{iter_number}.pkl", "wb") as chkpt_f:
        pickle.dump(checkpoint_data, chkpt_f)

    if str_to_bool(args.delete_prev_chkpt) and iter_number > 0:
        os.remove(checkpoint_dir + f"chkpt_{iter_number - 1}.pkl")

    if intermediate_result.fun < args.sufficient_cost:
        # Good enough for now
        raise StopIteration


result = minimize(
    objective.loss_function,
    aqc_initial_parameters,
    method=get_optimisation_method(),
    jac=True,
    options={
        "maxiter": args.max_iter,
        "ftol": np.nan,
        "gtol": np.sqrt(np.finfo(float).tiny),
    },
    callback=callback,
)

logger.info(result)

result_dir = "results/"
Path(result_dir).mkdir(parents=True, exist_ok=True)
result_data = {
    "args": args_dict,
    "result": result,
    "target_mps": target_mps,
    "aqc_ansatz": aqc_ansatz,
}

with open(os.path.join(result_dir, filename) + ".pkl", mode="wb") as file:
    pickle.dump(result_data, file)
