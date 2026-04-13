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
from tenpy.models.spins import SpinChain
from typing import List

logging.basicConfig()
logger = logging.getLogger(__name__)


class BondAlternatingModel(SpinChain):
    def __init__(self, L: int, J0: float, J1: float, hz: float | List = 0.0):
        """
        Class based on SpinChain with the bond-alternating Heisenberg Hamiltonian:

        H = sum_i{J_{i % 2} * (S^x_iS^x_{i+1} + S^y_iS^y_{i+1} + S^z_iS^z_{i+1})} - sum_i{hz_iS^z_i}.

        Args:
            L (int): The number of sites.
            J0 (float): The coupling strength between even-odd sites.
            J1 (float): The coupling strength between odd-even sites.
            hz (float | List): The external field strength (can be useful for breaking degeneracy).
        """
        # SpinChain doesn't work if J0 is close to 0.
        if abs(J0) < 1e-6:
            logger.warning(
                f"J0={J0} is too small, setting to 1e-6. Consider making J1=0 instead."
            )
            J0 = 1e-6

        # Build a SpinChain with Jx=Jy=Jz=J0
        super().__init__(
            dict(
                S=0.5,
                L=L,
                Jx=J0,
                Jy=J0,
                Jz=J0,
                hz=hz,
                bc_MPS="finite",
                conserve="None",
            )
        )

        # Set to True to suppress the warning when adding terms. We address the warning by calling
        # init_H_from_terms afterwards.
        self.manually_call_init_H = True

        # Construct the Hamiltonian.
        for i in range(1, L - 1, 2):
            # Apply (J1 - J0) * (S^x_iS^x_{i+1} + S^y_iS^y_{i+1} + S^z_iS^z_{i+1}) to odd bonds.
            # The overall coupling strength will be (J1 - J0) + J0 = J1
            for op in ["Sx", "Sy", "Sz"]:
                self.add_coupling_term(J1 - J0, i, i + 1, op, op)

        # Re-initialise H (addresses the warning TeNPy would have raised) and make sure the warning
        # will be raised if any extra terms are added in the future.
        self.init_H_from_terms()
        self.manually_call_init_H = False
