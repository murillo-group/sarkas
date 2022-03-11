"""Module of mathematical functions."""

import numpy as np
import numba as nb
from scipy.integrate import quad
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

    if potential_type in ["yukawa", "egs", "qsp", "hs_yukawa"]:
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


def inverse_fd_half(u):
    """Approximate the inverse of the Fermi-Dirac integral I_(-1/2)(u).

    function taken from Fukushima's code.

    Parameters
    ----------
    u : float
        I_(-1/2)(eta)

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
        difd1h = np.log(r)
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
        difd1h = np.sqrt(w)

    return difd1h


def fd_integral(eta, p):
    """Calculate the unnormalized Fermi-Dirac integral of order p with given chemical potential eta
    using scipy.integrate.quad.

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
    return quad(lambda x: x ** p / (1 + np.exp(x - eta)), 0, 100)[0]
