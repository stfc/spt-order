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
from scipy.optimize import OptimizeResult


def adam(
    fun,
    x0,
    jac,
    args=(),
    learning_rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    startiter=0,
    maxiter=1000,
    callback=None,
    **kwargs,
):
    """``scipy.optimize.minimize`` compatible implementation of ADAM -
    [http://arxiv.org/pdf/1412.6980.pdf].
    Adapted from ``autograd/misc/optimizers.py``.
    """
    x = x0
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for i in range(startiter, startiter + maxiter):
        g = jac(x)

        intermediate_result = OptimizeResult(
            x=x,
            fun=fun(x),
            jac=g,
            nit=i,
            nfev=i,
            success=True,
            message="Intermediate result",
        )
        if callback is not None:
            try:
                callback(intermediate_result)
            except StopIteration:
                return OptimizeResult(
                    x=x,
                    fun=fun(x),
                    jac=g,
                    nit=i,
                    nfev=i,
                    success=True,
                    status=99,
                    message="Desired cost reached early",
                )

        m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
        v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
        mhat = m / (1 - beta1 ** (i + 1))  # bias correction.
        vhat = v / (1 - beta2 ** (i + 1))
        x = x - learning_rate * mhat / (np.sqrt(vhat) + eps)

    return OptimizeResult(
        x=x,
        fun=fun(x),
        jac=g,
        nit=i + 1,
        nfev=i + 1,
        success=True,
        status=1,
        message="Max iterations reached",
    )
