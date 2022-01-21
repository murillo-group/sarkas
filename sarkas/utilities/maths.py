"""Module of mathematical functions."""

import numpy as np
import numba as nb
import scipy.signal as scp_signal

TWOPI = 2.0 * np.pi


def correlationfunction(At, Bt):
    """
    Calculate the correlation function between :math:`\\mathbf{A}(t)` and :math:`\\mathbf{B}(t)` using
    :func:`scipy.signal.correlate`

    .. math::
        C_{AB}(\\tau) =  \\sum_j^D \\sum_i^T A_j(t_i)B_j(t_i + \\tau)

    where :math:`D` is the number of dimensions and :math:`T` is the total length
    of the simulation.

    Parameters
    ----------
    At : numpy.ndarray
        Observable to correlate.

    Bt : numpy.ndarray
        Observable to correlate.

    Returns
    -------
    full_corr : numpy.ndarray
        Correlation function :math:`C_{AB}(\\tau)`
    """
    no_steps = At.size

    # Calculate the full correlation function.
    full_corr = scp_signal.correlate(At, Bt, mode="full")
    # Normalization of the full correlation function, Similar to norm_counter
    norm_corr = np.array([no_steps - ii for ii in range(no_steps)])
    # Find the mid point of the array
    mid = full_corr.size // 2
    # I want only the second half of the array, i.e. the positive lags only
    return full_corr[mid:] / norm_corr


@nb.njit
def fast_integral_loop(time, integrand):
    """Compute the integral with varying upper limit.

    Parameters
    ----------
    time: numpy.ndarray
        Domain of integration.

    integrand: numpy.ndarray
        Integrand.

    Returns
    -------
    integral : numpy.ndarray
        Integral with increasing upper limit. Shape = (time.len()).


    """
    integral = np.zeros_like(integrand)
    for it in range(1, len(time)):
        integral[it] = np.trapz(integrand[:it], x=time[:it])

    return integral


@nb.njit
def yukawa_green_function(k, alpha, kappa):
    """
    Evaluate the Green's function of Coulomb/Yukawa potential.

    .. math::

        G(k) = \\frac{4 \\pi }{\\kappa^2 + k^2} e^{- (k^2 + \\kappa^2)/4 \\alpha^2 }

    Parameters
    ----------
    k : numpy.ndarray, float
        Range or value at which to calculate the function.

    alpha: float
        Ewald screening parameter.

    kappa : float
        Inverse screening length.

    Returns
    -------
    _ : numpy.ndarray, float
        Green's function. See equation above

    """
    return 4.0 * np.pi * np.exp(-(k ** 2 + kappa ** 2) / (2 * alpha) ** 2) / (kappa ** 2 + k ** 2)


@nb.njit
def betamp(m, p, alpha, kappa):
    """
    Calculate the integral of the Yukawa Green's function

    .. math::

        \\beta(p,m) = \\int_0^\\infty dk \\, G_k^2 k^{2 (p + m + 2)}.

    See eq.(37) in :cite:`Dharuman2017`.
    """
    xa = np.linspace(0.0001, 500, 5000)
    Gk = yukawa_green_function(xa, alpha, kappa)
    return np.trapz(Gk * Gk * xa ** (2 * (m + p + 2)), x=xa)


@nb.njit
def force_error_approx_pppm(kappa, rc, p, h, alpha):
    """
    Calculate the total force error for a given value of the PPPM parameters.

    Parameters
    ----------
    kappa : float
        Inverse screening length.

    rc : float
        Cutoff length.

    p : int
        Charge assignment order.

    h : float
        Distance between two mesh points. Same for all directions.

    alpha : float
        Ewald screening parameter.

    Returns
    -------

    Tot_Delta_F : float
        Total force error given by

        .. math::
            \\Delta F = \\sqrt{\\Delta F_{\\textrm {pp}}^2 + \\Delta F_{\\textrm {pm}}^2 }

    pp_force_error : float
        PP force error.

    pm_force_error: float
        PM force error.

    """
    # Coefficients from :cite:`Deserno1998`
    if p == 1:
        Cmp = np.array([2 / 3])
    elif p == 2:
        Cmp = np.array([2 / 45, 8 / 189])
    elif p == 3:
        Cmp = np.array([4 / 495, 2 / 225, 8 / 1485])
    elif p == 4:
        Cmp = np.array([2 / 4725, 16 / 10395, 5528 / 3869775, 32 / 42525])
    elif p == 5:
        Cmp = np.array([4 / 93555, 2764 / 11609325, 8 / 25515, 7234 / 32531625, 350936 / 3206852775])
    elif p == 6:
        Cmp = np.array(
            [
                2764 / 638512875,
                16 / 467775,
                7234 / 119282625,
                1403744 / 25196700375,
                1396888 / 40521009375,
                2485856 / 152506344375,
            ]
        )
    elif p == 7:
        Cmp = np.array(
            [
                8 / 18243225,
                7234 / 1550674125,
                701872 / 65511420975,
                2793776 / 225759909375,
                1242928 / 132172165125,
                1890912728 / 352985880121875,
                21053792 / 8533724574375,
            ]
        )

    somma = 0.0
    for m in np.arange(p):
        expp = 2 * (m + p)
        somma += Cmp[m] * (2 / (1 + expp)) * betamp(m, p, alpha, kappa) * (h / 2.0) ** expp
    # eq.(36) in :cite:`Dharuman2017`
    pm_force_error = np.sqrt(3.0 * somma) / (2.0 * np.pi)

    # eq.(30) from :cite:`Dharuman2017`
    pp_force_error = 2.0 * np.exp(-((0.5 * kappa / alpha) ** 2) - alpha ** 2 * rc ** 2) / np.sqrt(rc)
    # eq.(42) from :cite:`Dharuman2017`
    Tot_DeltaF = np.sqrt(pm_force_error ** 2 + pp_force_error ** 2)

    return Tot_DeltaF, pp_force_error, pm_force_error


@nb.njit
def force_error_analytic_pp(potential_type, cutoff_length, potential_matrix, rescaling_const):
    """
    Calculate the force error from its analytic formula.

    Parameters
    ----------
    potential_type: str
        Type of potential used.

    potential_matrix: numpy.ndarray
        Potential parameters.

    cutoff_length : float, numpy.ndarray
        Cutoff radius of the potential.

    rescaling_const: float
        Constant by which to rescale the force error.

    Returns
    -------
    force_error: float
        Force error in units of Q^2/(4pi eps0) 1/a^2.

    """

    if potential_type in ["yukawa", "egs", "qsp"]:
        force_error = np.sqrt(TWOPI * potential_matrix[1, 0, 0]) * np.exp(-cutoff_length * potential_matrix[1, 0, 0])
    elif potential_type == "moliere":
        # Choose the smallest screening length for force error calculation

        force_error = np.sqrt(TWOPI * potential_matrix[:, 0, 0].min()) * np.exp(
            -cutoff_length * potential_matrix[1, 0, 0]
        )

    elif potential_type == "lj":
        # choose the highest sigma in case of multispecies
        sigma = potential_matrix[1, :, :].max()
        high_pow = potential_matrix[2, 0, 0]
        exponent = 2 * high_pow - 1
        force_error_tmp = high_pow ** 2 * sigma ** (2 * high_pow) / cutoff_length ** exponent
        force_error_tmp /= exponent
        force_error = np.sqrt(force_error_tmp)

    # Renormalize
    force_error *= rescaling_const

    return force_error
