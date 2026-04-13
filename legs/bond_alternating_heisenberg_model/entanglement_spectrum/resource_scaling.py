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


from qiskit.quantum_info import SparsePauliOp
import numpy as np
import matplotlib.pyplot as plt

plt.rc("font", family="serif")
plt.rcParams.update({"font.size": 12})
plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

string_list = ["I", "X", "Y", "Z"]

n_list = []
num_strings = []
num_commuting_groups = []

for n in range(1, 7):
    n_list.append(n)
    num_strings.append(4**n)

    characters = [None] * 4**n

    for number in range(4**n):
        if number == 0:
            string = "I" * n
        else:
            # Count in base-4.
            pad = int(n - 1 - np.floor(np.log(number) / np.log(4)))
            indices = str(np.base_repr(number, 4, pad))
            string = "".join([string_list[int(index)] for index in indices])

        characters[number] = string

    # Define an observable as the sum of all Pauli strings, so we can use `.group_commuting()` to
    # Find the number of qubit-wise commuting groups.
    obs = SparsePauliOp.from_list([(s, 1) for s in characters])

    commuting_groups = len(obs.group_commuting())
    num_commuting_groups.append(commuting_groups)
    print(f"n={n}, {4**n} strings, {commuting_groups} commuting groups")

plt.plot(
    n_list, num_strings, marker="x", label=r"Total number of Pauli strings ($4^l$)"
)
plt.plot(
    n_list,
    num_commuting_groups,
    marker="x",
    label=r"Number of qubit-wise commuting groups",
)
plt.xlabel(r"Segment length ($l$)")
plt.ylabel(r"Count")
plt.yscale("log")
plt.grid(which="both")
plt.legend()
plt.show()
