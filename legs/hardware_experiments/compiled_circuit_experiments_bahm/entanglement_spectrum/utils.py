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


import numpy as np
from typing import List, Tuple
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import eigh


def construct_density_matrix(
    l: int, observables: List[SparsePauliOp], exp_vals: np.ndarray
) -> np.ndarray:
    """
    Function to construct an l-site reduced density matrix (RDM) from the expectation value of Pauli
    strings:

    RDM = 1 / 2**l * sum_S <S> S, where S runs over every l-site Pauli string.

    Args:
        l (int): The number of sites.
        observables (List[SparsePauliOp]): A list of observables, with corresponding expectation
            values in `exp_vals` (in the same order).
        exp_vals (np.ndarray): The expectation values of the observables in `observables`.
    Returns:
        rdm (np.ndarray): The l-site RDM.
    """

    # Define Pauli operators
    operator_dict = {
        "I": np.array([[1, 0], [0, 1]]),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.array([[1, 0], [0, -1]]),
    }

    rdm = np.zeros((2**l, 2**l), dtype=complex)

    for i in range(len(observables)):
        # Convert SparsePauliOp to a list (of one element): (pauli, qubits, coeff)
        obs = observables[i]
        assert len(obs.to_sparse_list()) == 1

        term = obs.to_sparse_list()[0]
        pauli_list = ["I"] * l
        # Pauli string (doesn't include "I")
        pauli = term[0]
        # The ith character in `pauli` acts on qubit `qubits[i]`
        qubits = term[1]

        if pauli != "":

            for op, qubit in zip(pauli, qubits):
                pauli_list[qubit] = op

        # Take the tensor product of the terms in `pauli_list`
        matrix = operator_dict[pauli_list[0]]
        for k in range(1, len(pauli_list)):
            matrix = np.kron(matrix, operator_dict[pauli_list[k]])

        rdm += exp_vals[i] * matrix

    rdm /= 2**l

    return rdm


def find_eigenvalues(
    l: int,
    observables: List[SparsePauliOp],
    exp_vals: np.ndarray,
    stds: np.ndarray = None,
    bootstrap_samples: int = None,
) -> Tuple[np.ndarray | None]:
    """
    Function to calculate the eigenvalues of the reduced density matrix, given the expectation
    values of Pauli strings. If standard deviation are provided, the uncertainties in the
    eigenvalues are estimated via bootstrapping.

    RDM = 1 / 2**l * sum_S <S> S, where S runs over every l-site Pauli string.

    Bootstrapping: Given a number of samples (`bootstrap_samples` := n):
        1. Sample n instances of the expectation value of each Pauli string, drawn from a normal
            distribution with mean and standard deviation given by the corresponding entry of
            `exp_vals` and `stds`.
        2. Compute the RDM and transform it into the eigenbasis of the average RDM.
        3. Take the real part of the diagonal as the "eigenvalues". NOTE: These are not the true
            eigenvalues of the sampled RDM. We do this to enforce a consistent basis, and eliminate
            bias which would be introduced if we took the real eigenvalues and ordered them in
            descending order.
        3. Compute the mean and standard deviation of the eigenvalues

    NOTE: If no `stds` are provided, `eig_vals_mean` and `eig_vals_stds` are returned as None.

    Args:
        l (int): The number of sites.
        observables (List[SparsePauliOp]): A list of observables, the expectation values and errors
            of which are contained in `exp_vals` and `stds` (in the same order).
        exp_vals (np.ndarray): Expectation values of the observables in `observables`.
        stds (np.ndarray): Standard deviations of the observables in `observables`.
        bootstrap_samples (int): The number of samples to take when bootstrapping.
    Returns:
        eig_vals (np.ndarray): The RDM eigenvalues calculated from the Qiskit expectation values.
        eig_vals_mean (np.ndarray | None): The mean RDM eigenvalues calculated via sampling from
                                            the expectation value distribution.
        eig_vals_stds (np.ndarray | None): The standard deviation of the RDM eigenvalues calculated
                                            via sampling from the expectation value distribution.
    """

    rdm = construct_density_matrix(l, observables, exp_vals)

    # Find the eigenvalues of the RDM (descending order) and the corresponding eigenvectors.
    eig_vals, eig_vecs = eigh(rdm)
    eig_vals = eig_vals[::-1]
    eig_vecs = eig_vecs[:, ::-1]

    if stds is not None:
        # For each `exp_val` and `std`, sample `bootstrap_samples` samples from a normal
        # distribution centred on `exp_val` with standard deviation `std`.
        eig_vals_array = np.zeros((2**l, bootstrap_samples))
        exp_val_samples = np.zeros((exp_vals.shape[0], bootstrap_samples))
        for i in range(exp_vals.shape[0]):
            exp_val_samples[i, :] = np.random.normal(
                exp_vals[i], stds[i], bootstrap_samples
            )

        # For each sample, construct the RDM and find its eigenvalues.
        for j in range(bootstrap_samples):
            sampled_rdm = construct_density_matrix(
                l, observables, exp_val_samples[:, j]
            )
            # Transform the sampled RDM into the eigenbasis of the average RDM and take the real
            # part of the diagonal.
            eig_vals_array[:, j] = np.real(
                np.diag(np.conj(np.transpose(eig_vecs)) @ sampled_rdm @ eig_vecs)
            )

        # Compute the mean and standard deviation of the eigenvalues.
        eig_vals_mean = np.mean(eig_vals_array, axis=1)
        eig_vals_stds = np.std(eig_vals_array, axis=1)
    else:
        eig_vals_mean = None
        eig_vals_stds = None

    return eig_vals, eig_vals_mean, eig_vals_stds
