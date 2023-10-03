"""Module of mathematical functions."""

import scipy.signal as scp_signal
from numba import njit
from numpy import arange, array, exp, inf, ndarray, pi, sqrt, trapz, zeros_like
from scipy.integrate import quad

TWOPI = 2.0 * pi


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

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace( 0.0, 6.0 * np.pi, 3000)
    >>> w0 = 0.5
    >>> At = np.cos(w0 * t)
    >>> Bt = np.sin(w0 * t)
    >>> corr_t = correlationfunction(At, Bt)

    """
    no_steps = At.size

    # Calculate the full correlation function.
    full_corr = scp_signal.correlate(At, Bt, mode="full")
    # Normalization of the full correlation function, Similar to norm_counter
    norm_corr = array([no_steps - ii for ii in range(no_steps)])
    # Find the mid point of the array
    mid = full_corr.size // 2
    # I want only the second half of the array, i.e. the positive lags only
    return full_corr[mid:] / norm_corr


@njit
def fast_integral_loop(time, integrand):
    """Numba'd function to compute the following integral with a varying upper limit

    .. math::
        I(\\tau) = \\int_0^{\\tau} f(t) dt

    It uses :func:`numpy.trapz`. This function is used in the calculation of the transport coefficients.

    Parameters
    ----------
    time: numpy.ndarray
        Domain of integration

    integrand: numpy.ndarray
        Integrand.

    Returns
    -------
    integral : numpy.ndarray
        Integral with increasing upper limit. Shape = (time.len()).


    """
    integral = zeros_like(integrand)
    for it in range(1, len(time)):
        integral[it] = trapz(integrand[:it], x=time[:it])

    return integral


def yukawa_green_function(k: float, alpha: float, kappa: float):
    """
    Evaluate the Green's function of Coulomb/Yukawa potential.

    .. math::

        G(k) = \\frac{4 \\pi }{\\kappa^2 + k^2} e^{- (k^2 + \\kappa^2)/4 \\alpha^2 }

    Parameters
    ----------
    k : float, numpy.ndarray
        Range or value at which to calculate the function.

    alpha : float
        Ewald screening parameter.

    kappa : float
        Inverse screening length.

    Returns
    -------
    _ : numpy.ndarray, float
        Green's function. See equation above

    Examples
    --------
    >>> import numpy as np
    >>> k = np.linspace(0, 2, 100)
    >>> alpha = 0.2
    >>> kappa = 0.5
    >>> G_k = yukawa_green_function(k = k, alpha = alpha, kappa = kappa)

    """
    return 4.0 * pi * exp(-(k**2 + kappa**2) / (2 * alpha) ** 2) / (kappa**2 + k**2)


def betamp(m: int, p: int, alpha: float, kappa: float):
    """
    Calculate the integral of the Yukawa Green's function using :func:`scipy.integrate.quad`.

    .. math::

        \\beta(p,m) = \\int_0^\\infty dk \\, G_k^2 k^{2 (p + m + 2)}.

    See eq.(37) in :cite:`Dharuman2017`.

    Parameters
    ----------
    m : int
        :math:`m` index of the sum.

    p : int
        :math:`p` index of the sum

    alpha : float
        Ewald screening parameter :math:`\\alpha`.

    kappa : float
        Inverse screening length.

    Returns
    -------
    intgrl : float
        The integral :math:`\\beta(m,p)`.

    """
    exponent = 2 * (m + p + 2)
    intgrl, _ = quad(lambda x: yukawa_green_function(x, alpha, kappa) ** 2 * x ** (exponent), 0, inf)
    return intgrl


def force_error_approx_pppm(potential):
    r"""
    Calculates the force error, :math:`\Delta F_{\rm {pm}}`, for the PPPM algorithm using approximations given in :cite:`Dharuman2017`.
    The formula for :math:`\Delta F_{\rm {pm}}` can be found in :ref:`force_error`.

    Parameters
    ----------
    potential: :class:`sarkas.potentials.core.Potential`
       Potential class with all the required information.

    Returns
    -------
    tot_force_error: float
       Total force error given by the L2 norm of the PP and PM force errors.

    pppm_pm_err: float
        PM force error.

    pppm_pp_err: float
        PM force error.
    """

    if potential.type == "yukawa":
        kappa = potential.a_ws / potential.screening_length
        alpha = potential.pppm_alpha_ewald * potential.a_ws
        ha = potential.pppm_h_array[0] / potential.a_ws
        pppm_pm_err = force_error_approx_pm(kappa, potential.pppm_cao[0], ha, alpha)

    elif potential.type in ["coulomb", "qsp"]:
        alpha = potential.pppm_alpha_ewald * potential.a_ws
        ha = potential.pppm_h_array[0] / potential.a_ws
        pppm_pm_err = force_error_approx_pm(0.0, potential.pppm_cao[0], ha, alpha)

    rescaling_constant = sqrt(potential.total_num_density) * potential.a_ws**2
    pppm_pp_err = force_error_analytic_pp(
        potential.type, potential.rc, potential.screening_length, potential.pppm_alpha_ewald, rescaling_constant
    )
    pppm_pm_err *= sqrt(potential.total_num_density * potential.a_ws**3)
    force_error_tot = sqrt(pppm_pm_err**2 + pppm_pp_err**2)

    return force_error_tot, pppm_pm_err, pppm_pp_err


def force_error_approx_pm(kappa: float, p: int, h: float, alpha: float):
    r"""
    Calculates the PM part of the force error, :math:`\Delta F_{\rm {pm}}`,  for a given value of the PPPM parameters.
    The formula for :math:`\Delta F_{\rm {pm}}` can be found in :ref:`force_error`.

    Parameters
    ----------
    kappa : float
        Inverse screening length.

    p : int
        Charge assignment order.

    h : float
        Distance between two mesh points. Same for all directions.

    alpha : float
        Ewald screening parameter.

    Returns
    -------
    pm_force_error: float
        PM force error.

    """
    # Coefficients from :cite:`Deserno1998`
    if p == 1:
        Cmp = array([2 / 3])
    elif p == 2:
        Cmp = array([2 / 45, 8 / 189])
    elif p == 3:
        Cmp = array([4 / 495, 2 / 225, 8 / 1485])
    elif p == 4:
        Cmp = array([2 / 4725, 16 / 10395, 5528 / 3869775, 32 / 42525])
    elif p == 5:
        Cmp = array([4 / 93555, 2764 / 11609325, 8 / 25515, 7234 / 32531625, 350936 / 3206852775])
    elif p == 6:
        Cmp = array(
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
        Cmp = array(
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
    for m in arange(p):
        expp = 2 * (m + p)
        somma += Cmp[m] * (2 / (1 + expp)) * betamp(m, p, alpha, kappa) * (h / 2.0) ** expp
    # eq.(36) in :cite:`Dharuman2017`
    pm_force_error = sqrt(3.0 * somma) / (2.0 * pi)

    return pm_force_error


def force_error_analytic_pp(
    potential_type: str, cutoff_length: float, screening_length: float, alpha_ewald: float, rescaling_const: float
):
    """
    Calculate the short-range part of the force error from the approximation formula given in :cite:`Dharuman2017`.

    Parameters
    ----------
    potential_type: str
        Choice of potential.

    cutoff_length: float
        Short range cutoff.

    screening_length: float
        Screening length in case of screened potentials like yukawa. Pass 0 if coulomb

    alpha_ewald: float
        Ewald screening parameter.

    rescaling_const: float
        Constant by which to rescale the force error. \n
        In case of electric forces = :math:`Q^2/(4 \\pi \\epsilon_0) 1/a^2`.

    Returns
    -------
    pppm_pp_err: float
        Short range force error in units of `rescaling_const`.

    """
    if potential_type in ["yukawa"]:
        kappa = 1 / screening_length

        # PP force error calculation. Note that the equation was derived for a single component plasma.
        kappa_over_alpha = -0.25 * (kappa / alpha_ewald) ** 2
        alpha_times_rcut = -((alpha_ewald * cutoff_length) ** 2)
        # eq.(30) from :cite:`Dharuman2017`
        pppm_pp_err = 2.0 * exp(kappa_over_alpha + alpha_times_rcut) / sqrt(cutoff_length)
        # Renormalize
        pppm_pp_err *= rescaling_const

    elif potential_type in ["coulomb", "qsp"]:

        # PP force error calculation. Note that the equation was derived for a single component plasma.
        alpha_times_rcut = -((alpha_ewald * cutoff_length) ** 2)
        pppm_pp_err = 2.0 * exp(alpha_times_rcut) / sqrt(cutoff_length)
        # Renormalize
        pppm_pp_err *= rescaling_const

    return pppm_pp_err


def force_error_analytic_lcl(
    potential_type: str, cutoff_length: float, potential_matrix: ndarray, rescaling_const: float
):
    """
    Calculate the force error from its analytic formula.

    Parameters
    ----------
    potential_type : str
        Type of potential used.

    potential_matrix : numpy.ndarray
        Potential parameters.

    cutoff_length : float
        Cutoff radius of the potential.

    rescaling_const: float
        Constant by which to rescale the force error. \n
        In case of electric forces = :math:`Q^2/(4 \\pi \\epsilon_0) 1/a^2`.

    Returns
    -------
    force_error: float
        Force error in units of `rescaling_const`.


    Examples
    --------
    Yukawa, QSP, EGS force errors are calculated the same way.

    >>> import numpy as np
    >>> # Look more about potential matrix
    >>> potential_matrix = np.zeros((2, 2, 2))
    >>> kappa = 2.0 # in units of a_ws
    >>> potential_matrix[1,:,:] = kappa
    >>> rc = 6.0 # in units of a_ws
    >>> const = 1.0 # Rescaling const
    >>> force_error_analytic_lcl("yukawa", rc, potential_matrix, const)
    2.1780665692875655e-05

    Lennard jones potential

    >>> import numpy as np
    >>> potential_matrix = np.zeros( (5,2,2))
    >>> sigma = 3.4e-10
    >>> pot_const = 4.0 * 1.656e-21    # 4*epsilon
    >>> high_pow, low_pow = 12, 6
    >>> potential_matrix[0] = pot_const
    >>> potential_matrix[1] = sigma
    >>> potential_matrix[2] = high_pow
    >>> potential_matrix[3] = low_pow
    >>> rc = 10 * sigma
    >>> force_error_analytic_lcl("lj", rc, potential_matrix, 1.0)
    1.4590050212983888e-16

    Moliere potential

    >>> import numpy as np
    >>> from scipy.constants import epsilon_0, pi, elementary_charge
    >>> charge = 4.0 * elementary_charge  # = 4e [C] mks units
    >>> coul_const = 1.0 / (4.0 * pi * epsilon_0)
    >>> screening_charges = np.array([0.5, -0.5, 1.0])
    >>> screening_lengths = np.array([5.99988000e-11, 1.47732309e-11, 1.47732309e-11])  # [m]
    >>> params_len = len(screening_lengths)
    >>> pot_mat = np.zeros((2 * params_len + 1, 2, 2))
    >>> pot_mat[0] = coul_const * charge ** 2
    >>> pot_mat[1: params_len + 1] = screening_charges.reshape((3, 1, 1))
    >>> pot_mat[params_len + 1:] = 1. / screening_lengths.reshape((3, 1, 1))
    >>> rc = 6.629e-10
    >>> force_error_analytic_lcl("moliere", rc, pot_mat, 1.0)
    2.1223648580087958e-14

    """

    if potential_type in ["yukawa", "egs", "qsp", "hs_yukawa"]:
        force_error = sqrt(TWOPI * potential_matrix[1, 0, 0]) * exp(-cutoff_length * potential_matrix[1, 0, 0])
    elif potential_type == "moliere":
        # The first column of the potential matrix is 2*p + 1 long, where p is the
        # number of screening lengths.
        # The inverse of the screening_lengths is stored starting from p + 1.
        # Find p
        p = int((len(potential_matrix[:, 0, 0]) - 1) / 2)
        # Choose the smallest screening length ( i.e. max kappa) for force error calculation
        kappa = potential_matrix[p + 1 :, 0, 0].max()

        force_error = sqrt(TWOPI * kappa) * exp(-cutoff_length * kappa)

    elif potential_type == "lj":
        # choose the highest sigma in case of multispecies
        sigma = potential_matrix[1, :, :].max()
        high_pow = potential_matrix[2, 0, 0]
        exponent = 2 * high_pow - 1
        force_error_tmp = high_pow**2 * sigma ** (2 * high_pow) / cutoff_length**exponent
        force_error_tmp /= exponent
        force_error = sqrt(force_error_tmp)

    elif potential_type == "fitted":
        force_error = sqrt(TWOPI * potential_matrix[1, 0, 0]) * exp(-cutoff_length * potential_matrix[1, 0, 0])
    # Renormalize
    force_error *= rescaling_const

    return force_error
