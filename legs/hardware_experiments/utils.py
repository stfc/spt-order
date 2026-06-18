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
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import curve_fit
from typing import List, Tuple, Dict

plt.rc("font", family="serif")
plt.rc("text", usetex=True)


# Curve fitting
def exponential(x, a, b):
    return a * np.exp(-x / b)


def linear(x, a, b):
    return a + b * x


def polynomial_degree_2(x, a, b, c):
    return a + b * x + c * x**2


functions = {
    "exponential": exponential,
    "linear": linear,
    "polynomial_degree_2": polynomial_degree_2,
}

parameter_count = {
    "exponential": 2,
    "linear": 2,
    "polynomial_degree_2": 3,
}


def bayesian_information_criterion(n: int, ssr: float, k: int) -> float:
    """
    Function to calculate the Bayesian information criterion (BIC) for a given fit. The BIC is
    given by:

    BIC = n * ln(SSR / n) + k * ln(n)

    where n is the number of data points, SSR is the sum of the squared residuals between the raw
    and fitted data, and k is the number of parameters in the fitting function. The BIC aims to
    promote fits with lower SSR, whilst penalising fits with more parameters (to avoid overfitting),
    and a lower BIC indicates a better fit.

    Args:
        n (int): The number of data points.
        ssr (float): The sum of the squared residuals between the raw and fitted data.
        k (int): The number of parameters in the fitting function.
    Returns:
        (float): The BIC.
    """
    return n * np.log(ssr / n) + k * np.log(n)


def recompute_best_extrapolator(result_dict: Dict) -> Dict:
    """
    This function overwrites the "best extrapolator" expectation values
    (result_dict["PubResult.data"]["evs"]) and stds (result_dict["PubResult.data"]["stds"]). In
    these fields, Qiskit returns the first extrapolator which it deems to be successful, using some
    heuristic. This introduces a bias towards the first extrapolator (in our case exponential). We
    now use the extrapolator with the lowest Bayesian information criterion as the best fit, except
    for two exceptions. 1) if the range covered by the noisy data is below 0.1, we force a linear
    fit, and 2) if the extrapolator with the lowest BIC results in an unphysical extrapolation
    (>1 or <-1), we choose the next lowest BIC, and so on.

    Args:
        result_dict (dict): The results dictionary from submit_jobs.py.
    Returns:
        result_dict (dict): The modified results dictionary.
    """
    extrapolators = result_dict["PrimitiveResult.metadata"]["resilience"]["zne"][
        "extrapolator"
    ]

    for obs_index in range(
        result_dict["PubResult.metadata"]["resilience"]["zne"]["extrapolator"].shape[0]
    ):
        raw_data = result_dict["PubResult.data"]["evs_noise_factors"][obs_index]

        # Choose the fit with the lowest Bayesian information criterion.
        score = []
        for i in range(len(extrapolators)):

            fitted_data = result_dict["PubResult.data"]["evs_extrapolated"][obs_index][
                i
            ][1:]

            extrapolated_value = result_dict["PubResult.data"]["evs_extrapolated"][
                obs_index
            ][i][0]

            if extrapolated_value > 1 or extrapolated_value < -1:
                # If the EV is unphysical, do not choose.
                score.append(np.inf)
            elif (
                np.max(raw_data) - np.min(raw_data) < 0.1
                and extrapolators[i] == "linear"
            ):
                # If the variation is too low, choose linear.
                score.append(-np.inf)
            else:
                # Choose the lowest BIC.
                score.append(
                    bayesian_information_criterion(
                        raw_data.shape[0],
                        np.sum(np.square(raw_data - fitted_data)),
                        parameter_count[extrapolators[i]],
                    )
                )

        best_index = score.index(min(score))
        result_dict["PubResult.metadata"]["resilience"]["zne"]["extrapolator"][
            obs_index
        ] = extrapolators[best_index]
        result_dict["PubResult.data"]["evs"][obs_index] = result_dict["PubResult.data"][
            "evs_extrapolated"
        ][obs_index][best_index][0]
        result_dict["PubResult.data"]["stds"][obs_index] = result_dict[
            "PubResult.data"
        ]["stds_extrapolated"][obs_index][best_index][0]

    return result_dict


def get_observable_results(
    obs_of_interest: List[SparsePauliOp], result_dict: Dict
) -> Dict:
    """
    Function which, given a list of SparsePauliOp observables, will extract all the relavant data
    from result_dict (from submit_jobs.py) corresponding to those observables. For example, if you
    had the observables [XX, IX, YZ, YY, ZI] and you wanted the expectation values, etc. only for
    obs_of_interest = [YZ, ZI], you could use:
        get_observable_results(obs_of_interest, result_dict).

    NOTE: This only returns the entries of result_dict["PubResult.data"] and
    result_dict["PubResult.metadata"]["resilience"]["zne"]["extrapolator"], since these are the only
    things in result_dict which have different values for different observables. For anything else,
    you can use result_dict.

    Args:
        obs_of_interest (List[SparsePauliOp]): The list of relevant observables.
        result_dict (Dict): The results dictionary to extract from.
    Returns:
        results (Dict): A dictionary containing parts of result_dict relevant to the observables.
    """

    all_obs = result_dict["observables"]

    # Find the index (in all_obs) of each observable in obs_of_interest.
    # E.g. all_obs = [A, B, C, D, E], obs_of_interest = [A, D] -> indices = [0, 3]
    indices = [all_obs.index(obs) for obs in obs_of_interest]

    results = {}

    # Add everything from PubResult.data in the range defined by `indices`
    for key, value in result_dict["PubResult.data"].items():
        results[key] = value[np.array(indices)]

    # Also save the choice of extrapolator.
    # Everything else in result_dict shouldn't depend on the observable.
    if "zne" in result_dict["PubResult.metadata"]["resilience"].keys():
        results["extrapolator"] = result_dict["PubResult.metadata"]["resilience"][
            "zne"
        ]["extrapolator"][np.array(indices)]

    # Save the original indices for reference.
    results["indices"] = indices

    # Save the original indices for reference.
    results["indices"] = indices

    return results


def calculate_correlators_and_errors(
    z_evs: np.ndarray,
    zz_evs: np.ndarray,
    z_stds: np.ndarray,
    zz_stds: np.ndarray,
    zz_observables_list: List[SparsePauliOp],
) -> Tuple[np.ndarray]:
    """
    Function to calculate the correlators: <S^z_i S^z_j> - <S^z_i> * <S^z_j> and their corresponding
    errors, given the <Z_i> and <Z_iZ_j> values with their corresponding errors, and a list of
    SparsePauliOp, such that element i of that list is the observable whose expectation value is
    element i of zz_evs.

    Args:
        z_evs (np.ndarray): An array of <Z_i> values.
        zz_evs (np.ndarray): An array of <Z_iZ_j> values.
        z_stds (np.ndarray): The errors for z_evs.
        zz_stds (np.ndarray): The errors for zz_evs.
        zz_observables_list (List): A list of SparsePauliOp, such that element i of that list is the
                                observable whose expectation value is element i of zz_evs.
    Returns:
        correlators
    """
    n = z_evs.shape[0]
    correlators = np.empty((n, n))
    correlators[:] = np.nan
    correlators_err = np.empty((n, n))
    correlators_err[:] = np.nan

    for index in range(len(zz_evs)):
        pauli_string = zz_observables_list[index].paulis[0].to_label()
        indices = []
        for char_index in range(len(pauli_string)):
            if pauli_string[char_index] == "Z":
                indices.append(n - 1 - char_index)

        i = indices[0]
        j = indices[1]

        correlator = (zz_evs[index] - z_evs[i] * z_evs[j]) / 4
        # Error calculated using:
        #   e_{a+b} = sqrt((e_a)^2 + (e_b)^2)
        #   e_{a*b} = ab * sqrt((e_a/a)^2 + (e_b/b)^2)
        correlator_err = (
            np.sqrt(
                zz_stds[index] ** 2
                + z_evs[i] ** 2
                * z_evs[j] ** 2
                * ((z_stds[i] / z_evs[i]) ** 2 + (z_stds[j] / z_evs[j]) ** 2)
            )
            / 4
        )

        correlators[i, j] = correlator
        correlators[j, i] = correlator
        correlators_err[i, j] = correlator_err
        correlators_err[j, i] = correlator_err

    return correlators, correlators_err


def process_results_xxz(result_dict: dict, n: int) -> dict:
    """
    Function to compute the individual S^z expectation values, staggered magnetisation,
    magnetisation, and correlators from the hardware result dictionary, along with their errors.

    Args:
        result_dict (dict): The results dictionary from compiled_circuit_experiments_xxz/submit_jobs.py
        n (int): The number of qubits.
    Returns:
        data (dict): A dictionary containing the data, with the structure:
                    {
                        "value": {
                            "observable": value,
                            ...
                        },
                        "error": {
                            "observable": error,
                            ...
                        }
                    }
    """
    # TODO: We may want to modify the plotting scripts to use get_observable_results, as for BAHM.
    # Hardware EVs: <Z_i> and <Z_i Z_j>
    z_evs = result_dict["PubResult.data"]["evs"][:n]
    z_stds = result_dict["PubResult.data"]["stds"][:n]
    zz_evs = result_dict["PubResult.data"]["evs"][n:]
    zz_stds = result_dict["PubResult.data"]["stds"][n:]

    # Staggered magnetisation: sum((-1)^i * <S^z_i>)
    # Error calculated using: e_{a+b} = sqrt((e_a)^2 + (e_b)^2)
    stag_mag = np.sum((z_evs / 2) * np.array([(-1) ** i for i in range(n)])) / n
    stag_mag_err = np.sqrt(np.sum(np.square(z_stds / 2))) / n

    # Magnetisation: sum(<S^z_i>)
    # Error calculated using: e_{a+b} = sqrt((e_a)^2 + (e_b)^2)
    mag = np.sum(z_evs / 2) / n
    mag_err = np.sqrt(np.sum(np.square(z_stds / 2))) / n

    # Correlators: <S^z_i S^z_j> - <S^z_i> * <S^z_j>
    correlators, correlators_err = calculate_correlators_and_errors(
        z_evs, zz_evs, z_stds, zz_stds, result_dict["observables"][n:]
    )

    data = {
        "value": {
            "z": z_evs,
            "stag_mag": stag_mag,
            "mag": mag,
            "correlators": correlators,
        },
        "error": {
            "z": z_stds,
            "stag_mag": stag_mag_err,
            "mag": mag_err,
            "correlators": correlators_err,
        },
    }

    return data


def plot_zne(
    result_dict: dict,
    indices: List[int],
    tenpy_values: np.ndarray = None,
    hlines: List[float] = None,
    scale: float = 1,
):
    """
    Function to plot ZNE curves for observables indexed from n_min (inclusive) to n_max (exclusive)
    in result_dict. If the TeNPy values are known, they can be passed as `tenpy_values`.

    NOTE: The TeNPy values should be ordered relative to the order specified by `indices` I.e. if
    there are 4 observables in result_dict: O_0, O_1, O_2 and O_3, and you want to plot ZNE for
    observables O_3 and O_1 (in that order), you would input the full result_dict, indices=[3, 1],
    tenpy_values=[O_3_val, O_1_val].

    NOTE: `scale` is used to scale the Qiskit EVs, e.g. if Qiskit returns <Z>, but you want <Z>/2,
    you can use scale=0.5. This does NOT scale the values in `tenpy_values` or `hlines`.

    Args:
        result_dict (dict): The results dictionary from submit_jobs.py. Values will be scaled by
                            `scale`.
        n_min (int): The lowest index (inclusive).
        n_max (int): The highest index (exclusive).
        tenpy_values (np.ndarray): The analytic EVs from TeNPy.
        hlines (List[float]): Horizontal lines to plot. Can be for known bounds.
        scale (float): Scale the qiskit EVs by this value.
    Returns:
        fig, axs of the figure.
    """
    grid_size = int(np.ceil(np.sqrt(len(indices))))
    plt.rcParams.update({"font.size": 8})
    fig, axs = plt.subplots(grid_size, grid_size)
    fig.set_figheight(2 * grid_size)
    fig.set_figwidth(2 * grid_size)

    noise_factors = result_dict["PrimitiveResult.metadata"]["resilience"]["zne"][
        "noise_factors"
    ]
    extrapolated_noise_factors = result_dict["PrimitiveResult.metadata"]["resilience"][
        "zne"
    ]["extrapolated_noise_factors"]

    for rel_index, obs_index in enumerate(indices):
        row = rel_index // grid_size
        col = rel_index % grid_size

        # Raw data
        evs_noise_factors = (
            result_dict["PubResult.data"]["evs_noise_factors"][obs_index] * scale
        )
        stds_noise_factors = (
            result_dict["PubResult.data"]["stds_noise_factors"][obs_index] * scale
        )

        chosen_extrapolator = result_dict["PubResult.metadata"]["resilience"]["zne"][
            "extrapolator"
        ][obs_index]

        # Qiskit fitted data
        evs_extrapolated = (
            result_dict["PubResult.data"]["evs_extrapolated"][obs_index] * scale
        )
        ev = result_dict["PubResult.data"]["evs"][obs_index] * scale
        std = result_dict["PubResult.data"]["stds"][obs_index] * scale

        try:
            func = functions[chosen_extrapolator]

            # My fit for the data
            popt, _ = curve_fit(func, noise_factors, evs_noise_factors)
            x = np.linspace(0, max(noise_factors), 100)
            y = func(x, *popt)
            axs[row, col].plot(x, y, alpha=0.5, color="k", linestyle="--")

            # Qiskit's fit
            index = result_dict["PrimitiveResult.metadata"]["resilience"]["zne"][
                "extrapolator"
            ].index(chosen_extrapolator)
            popt, _ = curve_fit(
                func, extrapolated_noise_factors, evs_extrapolated[index]
            )
            y = func(x, *popt)
            axs[row, col].plot(x, y, alpha=1.0)
        except:
            pass

        # Add extrapolated value at noise factor 0.
        evs_noise_factors = [ev] + list(evs_noise_factors)
        errors = [std] + list(stds_noise_factors)

        # Individual plots:
        axs[row, col].errorbar(
            [0.0] + noise_factors, evs_noise_factors, errors, fmt="x", alpha=1.0
        )
        if tenpy_values is not None:
            axs[row, col].scatter(0, tenpy_values[rel_index], color="tab:green")
        if hlines is not None:
            [axs[row, col].axhline(i, color="k", linestyle="--") for i in hlines]
        axs[row, col].axhline(0, color="k")
        axs[row, col].set_title(
            f"Qubit: {obs_index}\nExtrapolator: {chosen_extrapolator}"
        )

    return fig, axs
