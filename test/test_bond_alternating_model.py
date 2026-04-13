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
from legs.bond_alternating_heisenberg_model.model import BondAlternatingModel
from tenpy import MPS

# -------------
# --- Utils ---
# -------------


def construct_bond_alternating_hamiltonian(L: int, J0: float, J1: float) -> np.ndarray:
    """
    Generates the Hamiltonian for the bond-alternating Heisenberg model:

    H = sum_i{J_{i % 2} * (S^x_iS^x_{i+1} + S^y_iS^y_{i+1} + S^z_iS^z_{i+1})}.

    Args:
        L (int): The number of sites.
        J0 (float): The coupling strength between even-odd sites.
        J1 (float): The coupling strength between odd-even sites.
    """
    # 2x2 matrices, with Y and Z defined as for TeNPy SpinSite (swapped basis)
    I = np.eye(2, 2)
    X = np.array(
        [
            [0, 1],
            [1, 0],
        ]
    )
    Y = np.array(
        [
            [0, 1j],
            [-1j, 0],
        ]
    )
    Z = np.array(
        [
            [-1, 0],
            [0, 1],
        ]
    )

    # Construct Hamiltonian matrix.
    H = np.zeros((2**L, 2**L), dtype=np.complex128)

    for i in range(0, L - 1):
        coeff = (J0 if i % 2 == 0 else J1) / 4
        for op in [X, Y, Z]:
            op_list = [op if (j == i) or (j == (i + 1)) else I for j in range(L)]
            op_mat = op_list[0]
            for j in range(1, L):
                op_mat = np.kron(op_mat, op_list[j])

            # op_mat is the matrix form of I...IXXI...I (or Y or Z) in big-endian notation (as TeNPy
            # uses).
            H += coeff * op_mat

    return H


# -------------
# --- Tests ---
# -------------


def test_zero_state_energy():
    """
    Checks that the energy of the |0...0> state is as expected.
    """
    L = np.random.randint(2, 200)
    # Random couplings between -10 and 10
    J = (np.random.random(2) - 0.5 * np.ones(2)) * 10

    model = BondAlternatingModel(L, J[0], J[1])

    # There are L // 2 bonds with J0 coupling and (L - 1) // 2 bonds with J1 coupling.
    # The |000...> state is an eigenstate with energy:
    # (J0 / 4)(L // 2) + (J1 / 4)((L - 1) // 2).

    psi = MPS.from_product_state(model.lat.mps_sites(), ["up"] * L, bc="finite")

    energy = model.calc_H_MPO().expectation_value(psi)
    expected_energy = (J[0] / 4) * (L // 2) + (J[1] / 4) * ((L - 1) // 2)

    np.testing.assert_almost_equal(energy, expected_energy, decimal=12)


def test_hamiltonian_terms():
    """
    Checks that the terms in the Hamiltonian are as expected. I.e. J0/4 X_0 X_1.
    """
    L = np.random.randint(2, 200)
    # Random couplings between -10 and 10
    J = (np.random.random(2) - 0.5 * np.ones(2)) * 10

    model = BondAlternatingModel(L, J[0], J[1])

    # Find all the "terms" in the model Hamiltonian, and group them. I.e. group:
    # a*X0X1 + b*X0X1 into (a + b)*X0X1
    term_list = model.calc_H_MPO().to_TermList(["Id", "Sx", "Sy", "Sz"])
    term_list.order_combine(model.lat.mps_sites())

    unique_terms = {}
    for term, strength in zip(term_list.terms, term_list.strength):
        if str(term) not in unique_terms.keys():
            unique_terms[str(term)] = strength
        else:
            unique_terms[str(term)] += strength

    # Define all the terms we expect to be in the Hamiltonian. I.e. J0*X0X1.
    expected_terms = {}
    for i in range(L - 1):
        for op in ["Sx", "Sy", "Sz"]:
            expected_terms[str([(op, i), (op, i + 1)])] = J[i % 2] + 0j

    # For all the terms in expected_terms, check the term exists in unique_terms, and the strength
    # is correct.
    for key, value in expected_terms.items():
        assert key in unique_terms.keys()
        np.testing.assert_almost_equal(unique_terms[key], value, decimal=12)

    # For all the terms in unique_terms:
    #   - if the term is in expected_terms, check the strength is correct.
    #   - if the term is not expected, check the strength is 0.
    for key, value in unique_terms.items():
        if key in expected_terms.keys():
            np.testing.assert_almost_equal(expected_terms[key], value, decimal=12)
        else:
            np.testing.assert_almost_equal(0, value, decimal=12)

    # This means that all expected terms are included, and any unexpected terms have strength 0.


def test_hamiltonian_matrix_form():
    """
    Checks that the 2^L x 2^L matrix form of the Hamiltonian is as expected.
    """
    L = np.random.randint(2, 10)
    # Random couplings between -10 and 10
    J = (np.random.random(2) - 0.5 * np.ones(2)) * 10

    # Get Hamiltonian matrix
    H = construct_bond_alternating_hamiltonian(L, J[0], J[1])

    # Get Hamiltonian from BondAlternatingModel
    model = BondAlternatingModel(L, J[0], J[1])

    # MPO form of the Hamiltonian
    H_MPO = model.calc_H_MPO()

    # Extract the W tensors
    tensors = [
        H_MPO.get_W(i, copy=True).itranspose(["p", "p*", "wL", "wR"]).to_ndarray()
        for i in range(L)
    ]

    # Contract wR leg of tensor i with wL leg of tensor i+1
    contraction = tensors[0]
    for i in range(1, L):
        contraction = np.tensordot(contraction, tensors[i], [[-1], [2]])

    # Take the 0 element of left-virtual leg wL_0 and the 1 element of the right-virtual leg wL_L.
    # NOTE: I'm not sure why you pick these specific ones. If you take 0,0 or 1,1, you get the
    # identity, and if you take 1,0, you get the zero matrix. Taking 0,1 gives the correct
    # Hamiltonian.
    contraction = np.take(contraction, 0, 2)
    contraction = np.take(contraction, 1, -1)

    # The resulting matrix has indices: p_0, p*_0, ..., p_L, p*_L.
    # Swap to the order: p*_0, ..., p*_L, p_0, ..., p_L (H maps from p to p*)
    perm = [2 * i + 1 for i in range(L)] + [2 * i for i in range(L)]
    hamiltonian = np.transpose(contraction, perm).reshape((2**L, 2**L))

    np.testing.assert_array_almost_equal(hamiltonian, H, decimal=12)


def test_no_warning_when_init():
    """
    Checks that the warnings raised when adding extra terms to the model are suppressed during
    initialisation.
    """
    with np.testing.assert_no_warnings():
        BondAlternatingModel(10, 1.0, 0.5)


def test_warning_when_add_terms():
    """
    Checks that the suppression of warnings is removed after the model is initialised.
    """
    model = BondAlternatingModel(10, 1.0, 0.5)

    with np.testing.assert_warns(UserWarning):
        model.add_onsite_term(0.5, 0, "Sz")


def test_H_reinitialised():
    """
    Checks that the Hamiltonian is re-initialised after the extra terms have been added. I.e. checks
    that we addressed the warning TeNPy raised (which we suppressed).
    """
    model = BondAlternatingModel(10, 1.0, 0.5)

    H_MPO_before = model.H_MPO

    model.init_H_from_terms()

    H_MPO_after = model.H_MPO

    assert H_MPO_before.is_equal(H_MPO_after)
