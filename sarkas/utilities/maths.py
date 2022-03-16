"""Module of mathematical functions."""

from numpy import pi, exp, trapz, sqrt, log, array, zeros_like, inf, arange
from numba import njit
from scipy.integrate import quad
import scipy.signal as scp_signal

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


def yukawa_green_function(k: float, alpha: float, kappa: float) -> float:
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
    return 4.0 * pi * exp(-(k ** 2 + kappa ** 2) / (2 * alpha) ** 2) / (kappa ** 2 + k ** 2)


def betamp(m: int, p: int, alpha: float, kappa: float) -> float:
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


def force_error_approx_pppm(kappa, rc, p, h, alpha):
    """
    Numba'd function to calculate the total force error for a given value of the PPPM parameters.

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

    # eq.(30) from :cite:`Dharuman2017`
    pp_force_error = 2.0 * exp(-((0.5 * kappa / alpha) ** 2) - alpha ** 2 * rc ** 2) / sqrt(rc)
    # eq.(42) from :cite:`Dharuman2017`
    Tot_DeltaF = sqrt(pm_force_error ** 2 + pp_force_error ** 2)

    return Tot_DeltaF, pp_force_error, pm_force_error


def force_error_analytic_pp(potential_type, cutoff_length, potential_matrix, rescaling_const):
    """
    Calculate the force error from its analytic formula.

    Parameters
    ----------
    potential_type : str
        Type of potential used.

    potential_matrix : numpy.ndarray
        Potential parameters.

    cutoff_length : numpy.ndarray
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
    >>> force_error_analytic_pp("yukawa", rc, potential_matrix, const)
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
    >>> force_error_analytic_pp("lj", rc, potential_matrix, 1.0)
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
    >>> force_error_analytic_pp("moliere", rc, pot_mat, 1.0)
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
        kappa = potential_matrix[p + 1:, 0, 0].max()

        force_error = sqrt(TWOPI * kappa) * exp(- cutoff_length * kappa)

    elif potential_type == "lj":
        # choose the highest sigma in case of multispecies
        sigma = potential_matrix[1, :, :].max()
        high_pow = potential_matrix[2, 0, 0]
        exponent = 2 * high_pow - 1
        force_error_tmp = high_pow ** 2 * sigma ** (2 * high_pow) / cutoff_length ** exponent
        force_error_tmp /= exponent
        force_error = sqrt(force_error_tmp)

    # Renormalize
    force_error *= rescaling_const

    return force_error


def inverse_fd_half(u: float) -> float:
    """Approximate the inverse of the Fermi-Dirac integral :math:`I_{-1/2}(\\eta)` using the fits provided by Fukushima.
    Function translated from Fukushima's code.
    TODO: Add references.

    Parameters
    ----------
    u : float
        Normalized electron density :math:` = \\Lambda_{\\rm deB}^3 n_e \\sqrt{\\pi}/4`, \n
        where :math:`\\Lambda_{\\rm deB}` is the de Broglie wavelength of the electron gas.

    Examples
    --------
    >>> import numpy as np
    >>> # Values taken from Tutorial Notebooks
    >>> ne = 1.62e32 # [N/m^3]
    >>> lambda_deB = 1.957093e-11 # [m]
    >>> u = lambda_deB**3 * ne * np.sqrt(np.pi)/4.0
    >>> eta = inverse_fd_half(u)
    >>> f"{eta:.4f}"
    '-0.2860'

    Returns
    -------
    difdih : float
        Scaled chemical potential.

    """
    u1 = 5.8936762325936050502
    u2 = 20.292139527268839575
    u3 = 69.680386701202787485
    u4 = 246.24741525281437791
    vc = 1543.4606939459401869

    if u < u1:
        v = u1 - u
        r = u * (4.4225399543845577739e9
                 + u * (+1.4318826531216930391e9
                        + u * (+2.0024511162084252731e8
                               + u * (+1.5771885953346837109e7
                                      + u * (+7.664073281017674960e5
                                             + u * (+2.3599362498847900809e4
                                                    + u * (+4.4114601741712557348e2
                                                           + u * (+3.8844451277821727933e0)))))))) \
            / (+2.448084356710615572e9
               + v * (+2.063907695769060888e8
                      + v * (+6.943821586626002503e6
                             + v * (+8.039397005856418743e4
                                    + v * (-1.791261676399435220e3
                                           + v * (-7.908051927048792349e1
                                                  - v))))))
        difd1h = log(r)
    elif u < u2:
        y = u - u1
        difd1h = \
            (+5.849893914158469793e14
             + y * (+3.353389340896418967e14
                    + y * (+7.300790845633384552e13
                           + y * (+7.531271098292146768e12
                                  + y * (+3.726221594134586141e11
                                         + y * (+7.827935737269045014e9
                                                + y * (+5.021972425404123509e7))))))) \
            / (+1.4397298668421281743e14
               + y * (+6.440007889067504875e13
                      + y * (+1.0498882082904393876e13
                             + y * (+7.568424788316453035e11
                                    + y * (+2.3206228235771973103e10
                                           + y * (+2.4329782896354397638e8
                                                  + y * (+3.6768664133860359837e5
                                                         + y * (-5.924317283823514482e2
                                                                + y))))))))
    elif u < u3:
        y = u - u2
        difd1h = \
            (+6.733834344762314874e18
             + y * (+1.1385291167086018856e18
                    + y * (+7.441797125810403052e16
                           + y * (+2.3556527595722738211e15
                                  + y * (+3.6904107711114070061e13
                                         + y * (+2.5927357055940595308e11
                                                + y * (+5.989403440741097470e8))))))) \
            / (+6.968777783221497285e17
               + y * (+9.451599633557071205e16
                      + y * (+4.7388759083089595117e15
                             + y * (+1.0766510215928549449e14
                                    + y * (+1.0888539870400255904e12
                                           + y * (+4.0374047390260294467e9
                                                  + y * (+2.3126814357531839818e6
                                                         + y * (-1.4788294703774470115e3
                                                                + y))))))))
    elif u < u4:

        y = u - u3
        difd1h = \
            (+7.884494095314249799e19
             + y * (+3.7486465573810023777e18
                    + y * (+6.934193474730824900e16
                           + y * (+6.302949477641708425e14
                                  + y * (+2.9299316609051704688e12
                                         + y * (+6.591658047866512380e9
                                                + y * (+6.082995857672390394e6
                                                       + y * (+1.5054843420905807932e3)))))))) \
            / (+3.5593247304804720533e18
               + y * (+1.3505797700306451874e17
                      + y * (+1.9160919212553016350e15
                             + y * (+1.2652560651095328402e13
                                    + y * (+3.9491055033213850687e10
                                           + y * (+5.253083775042776560e7
                                                  + y * (+2.2252541165920236251e4
                                                         + y)))))))
    else:
        t = vc * (u ** -(4.0 / 3.0))
        s = 1.0 - t
        w = (+3.4330125059142833612e7
             + s * (+8.713462091032493289e5
                    + s * (+2.4245560148256419080e3
                           + s))) / \
            (t * (+1.2961677595919532465e4
                  + s * (+3.2092883892793106635e2
                         + s * (+0.7192193760323717351e0))))
        difd1h = sqrt(w)

    return difd1h


def fd_integral(eta: float, p: float) -> float:
    """Calculate the unnormalized Fermi-Dirac integral :math:`\\mathcal I_p(\\eta)` of order `p`
     with given chemical potential `eta`. It uses :func:`scipy.integrate.quad`. \n
     The integral is defined as

    .. math::
        \\mathcal I_{p} [\\eta] =  \\int_0^{\\infty} dx \\frac{x^p}{1 + e^{x - \\eta} }.

    Examples
    --------
    >>> from numpy import pi, sqrt
    >>> # Electron density
    >>> eta = -0.2860
    >>> I = fd_integral(eta = eta, p = 0.5)
    >>> lambda_deB = 1.957093e-11 # [m]
    >>> ne = 4.0/( sqrt(pi) * lambda_deB**3 ) * I
    >>> f"{ne:.4e}"
    '1.6201e+32'

    Parameters
    ----------
    eta : float
        Chemical potential divided by temperature.

    p : float
        Order of the FD integral.

    Returns
    -------

    _ : float
        FD Integral.

    """
    return quad(lambda x: x ** p / (1 + exp(x - eta)), 0, 100)[0]
