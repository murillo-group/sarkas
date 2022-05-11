"""Module of mathematical functions."""

import scipy.signal as scp_signal
from numba import njit
from numpy import arange, array, exp, inf, log, pi, sqrt, trapz, zeros_like
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
    return 4.0 * pi * exp(-(k**2 + kappa**2) / (2 * alpha) ** 2) / (kappa**2 + k**2)


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
    pp_force_error = 2.0 * exp(-((0.5 * kappa / alpha) ** 2) - alpha**2 * rc**2) / sqrt(rc)
    # eq.(42) from :cite:`Dharuman2017`
    Tot_DeltaF = sqrt(pm_force_error**2 + pp_force_error**2)

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
        r = (
            u
            * (
                4.4225399543845577739e9
                + u
                * (
                    +1.4318826531216930391e9
                    + u
                    * (
                        +2.0024511162084252731e8
                        + u
                        * (
                            +1.5771885953346837109e7
                            + u
                            * (
                                +7.664073281017674960e5
                                + u
                                * (
                                    +2.3599362498847900809e4
                                    + u * (+4.4114601741712557348e2 + u * (+3.8844451277821727933e0))
                                )
                            )
                        )
                    )
                )
            )
            / (
                +2.448084356710615572e9
                + v
                * (
                    +2.063907695769060888e8
                    + v
                    * (
                        +6.943821586626002503e6
                        + v
                        * (+8.039397005856418743e4 + v * (-1.791261676399435220e3 + v * (-7.908051927048792349e1 - v)))
                    )
                )
            )
        )
        difd1h = log(r)
    elif u < u2:
        y = u - u1
        difd1h = (
            +5.849893914158469793e14
            + y
            * (
                +3.353389340896418967e14
                + y
                * (
                    +7.300790845633384552e13
                    + y
                    * (
                        +7.531271098292146768e12
                        + y * (+3.726221594134586141e11 + y * (+7.827935737269045014e9 + y * (+5.021972425404123509e7)))
                    )
                )
            )
        ) / (
            +1.4397298668421281743e14
            + y
            * (
                +6.440007889067504875e13
                + y
                * (
                    +1.0498882082904393876e13
                    + y
                    * (
                        +7.568424788316453035e11
                        + y
                        * (
                            +2.3206228235771973103e10
                            + y
                            * (
                                +2.4329782896354397638e8
                                + y * (+3.6768664133860359837e5 + y * (-5.924317283823514482e2 + y))
                            )
                        )
                    )
                )
            )
        )
    elif u < u3:
        y = u - u2
        difd1h = (
            +6.733834344762314874e18
            + y
            * (
                +1.1385291167086018856e18
                + y
                * (
                    +7.441797125810403052e16
                    + y
                    * (
                        +2.3556527595722738211e15
                        + y
                        * (+3.6904107711114070061e13 + y * (+2.5927357055940595308e11 + y * (+5.989403440741097470e8)))
                    )
                )
            )
        ) / (
            +6.968777783221497285e17
            + y
            * (
                +9.451599633557071205e16
                + y
                * (
                    +4.7388759083089595117e15
                    + y
                    * (
                        +1.0766510215928549449e14
                        + y
                        * (
                            +1.0888539870400255904e12
                            + y
                            * (
                                +4.0374047390260294467e9
                                + y * (+2.3126814357531839818e6 + y * (-1.4788294703774470115e3 + y))
                            )
                        )
                    )
                )
            )
        )
    elif u < u4:

        y = u - u3
        difd1h = (
            +7.884494095314249799e19
            + y
            * (
                +3.7486465573810023777e18
                + y
                * (
                    +6.934193474730824900e16
                    + y
                    * (
                        +6.302949477641708425e14
                        + y
                        * (
                            +2.9299316609051704688e12
                            + y
                            * (+6.591658047866512380e9 + y * (+6.082995857672390394e6 + y * (+1.5054843420905807932e3)))
                        )
                    )
                )
            )
        ) / (
            +3.5593247304804720533e18
            + y
            * (
                +1.3505797700306451874e17
                + y
                * (
                    +1.9160919212553016350e15
                    + y
                    * (
                        +1.2652560651095328402e13
                        + y
                        * (+3.9491055033213850687e10 + y * (+5.253083775042776560e7 + y * (+2.2252541165920236251e4 + y)))
                    )
                )
            )
        )
    else:
        t = vc * (u ** -(4.0 / 3.0))
        s = 1.0 - t
        w = (+3.4330125059142833612e7 + s * (+8.713462091032493289e5 + s * (+2.4245560148256419080e3 + s))) / (
            t * (+1.2961677595919532465e4 + s * (+3.2092883892793106635e2 + s * (+0.7192193760323717351e0)))
        )
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
    return quad(lambda x: x**p / (1 + exp(x - eta)), 0, 100)[0]


def fdm1h(x: float) -> float:
    """
    double precision rational minimax approximation of Fermi-Dirac integral of order k=-1/2
    ! Reference: Fukushima, T. (2015) Appl. Math. Comp., 259, 708-729
    ! "Precise and fast computation of Fermi-Dirac integral of integer
    !  and half integer order by piecewise minimax rational approximation"
    !
    ! Author: Fukushima, T. <Toshio.Fukushima@nao.ac.jp>
    !
    """
    if x < -2.0:
        ex = exp(x)
        t = ex * 7.38905609893065023
        fd = ex * (
            1.77245385090551603
            - ex
            * (
                40641.4537510284430
                + t
                * (9395.7080940846442 + t * (649.96168315267301 + t * (12.7972295804758967 + t * 0.00153864350767585460)))
            )
            / (32427.1884765292940 + t * (11079.9205661274782 + t * (1322.96627001478859 + t * (63.738361029333467 + t))))
        )
    elif x < 0.0:
        s = -0.5 * x
        t = 1.0 - s
        fd = (
            272.770092131932696
            + t
            * (
                30.8845653844682850
                + t
                * (
                    -6.43537632380366113
                    + t
                    * (
                        14.8747473098217879
                        + t
                        * (
                            4.86928862842142635
                            + t
                            * (
                                -1.53265834550673654
                                + t * (-1.02698898315597491 + t * (-0.177686820928605932 - t * 0.00377141325509246441))
                            )
                        )
                    )
                )
            )
        ) / (
            293.075378187667857
            + s
            * (
                305.818162686270816
                + s
                * (
                    299.962395449297620
                    + s
                    * (
                        207.640834087494249
                        + s * (92.0384803181851755 + s * (37.0164914112791209 + s * (7.88500950271420583 + s)))
                    )
                )
            )
        )
    elif x < 2.0:
        t = 0.5 * x
        fd = (
            3531.50360568243046
            + t
            * (
                6077.5339658420037
                + t
                * (
                    6199.7700433981326
                    + t
                    * (
                        4412.78701919567594
                        + t
                        * (
                            2252.27343092810898
                            + t * (811.84098649224085 + t * (191.836401053637121 + t * 23.2881838959183802))
                        )
                    )
                )
            )
        ) / (
            3293.83702584796268
            + t
            * (
                1528.97474029789098
                + t
                * (
                    2568.48562814986046
                    + t
                    * (
                        925.64264653555825
                        + t * (574.23248354035988 + t * (132.803859320667262 + t * (29.8447166552102115 + t)))
                    )
                )
            )
        )
    elif x < 5.0:
        t = 0.3333333333333333333 * (x - 2.0)
        fd = (
            4060.70753404118265
            + t
            * (
                10812.7291333052766
                + t
                * (
                    13897.5649482242583
                    + t
                    * (
                        10628.4749852740029
                        + t
                        * (
                            5107.70670190679021
                            + t * (1540.84330126003381 + t * (284.452720112970331 + t * 29.5214417358484151))
                        )
                    )
                )
            )
        ) / (
            1564.58195612633534
            + t
            * (
                2825.75172277850406
                + t
                * (
                    3189.16066169981562
                    + t
                    * (
                        1955.03979069032571
                        + t * (828.000333691814748 + t * (181.498111089518376 + t * (32.0352857794803750 + t)))
                    )
                )
            )
        )
    elif x < 10.0:
        t = 0.2 * x - 1.0
        fd = (
            1198.41719029557508
            + t
            * (
                3263.51454554908654
                + t
                * (
                    3874.97588471376487
                    + t
                    * (
                        2623.13060317199813
                        + t
                        * (
                            1100.41355637121217
                            + t * (267.469532490503605 + t * (25.4207671812718340 + t * 0.389887754234555773))
                        )
                    )
                )
            )
        ) / (
            273.407957792556998
            + t
            * (
                595.918318952058643
                + t
                * (
                    605.202452261660849
                    + t * (343.183302735619981 + t * (122.187622015695729 + t * (20.9016359079855933 + t)))
                )
            )
        )
    elif x < 20.0:
        t = 0.1 * x - 1.0
        fd = (
            9446.00169435237637
            + t
            * (
                36843.4448474028632
                + t
                * (
                    63710.1115419926191
                    + t
                    * (
                        62985.2197361074768
                        + t
                        * (
                            37634.5231395700921
                            + t * (12810.9898627807754 + t * (1981.56896138920963 + t * 81.4930171897667580))
                        )
                    )
                )
            )
        ) / (
            1500.04697810133666
            + t
            * (
                5086.91381052794059
                + t
                * (
                    7730.01593747621895
                    + t
                    * (
                        6640.83376239360596
                        + t * (3338.99590300826393 + t * (860.499043886802984 + t * (78.8565824186926692 + t)))
                    )
                )
            )
        )
    elif x < 40.0:
        t = 0.05 * x - 1.0
        fd = (
            22977.9657855367223
            + t
            * (
                123416.616813887781
                + t
                * (
                    261153.765172355107
                    + t
                    * (
                        274618.894514095795
                        + t
                        * (
                            149710.718389924860
                            + t * (40129.3371700184546 + t * (4470.46495881415076 + t * 132.684346831002976))
                        )
                    )
                )
            )
        ) / (
            2571.68842525335676
            + t
            * (
                12521.4982290775358
                + t
                * (
                    23268.1574325055341
                    + t
                    * (
                        20477.2320119758141
                        + t * (8726.52577962268114 + t * (1647.42896896769909 + t * (106.475275142076623 + t)))
                    )
                )
            )
        )
    else:
        w = 1.0 / (x * x)
        t = 1600.0 * w
        fd = (
            sqrt(x)
            * 2.0
            * (
                1.0
                - w
                * (
                    0.411233516712009968
                    + t
                    * (
                        0.00110980410034088951
                        + t
                        * (
                            0.0000113689298990173683
                            + t * (2.56931790679436797e-7 + t * (9.97897786755446178e-9 + t * 8.67667698791108582e-10))
                        )
                    )
                )
            )
        )
    return fd


def fd1h(x: float) -> float:
    if x < -2.0:
        ex = exp(x)
        t = ex * 7.38905609893065023

        fd = ex * (
            0.886226925452758014
            - ex
            * (
                19894.4553386951666
                + t
                * (
                    4509.64329955948557
                    + t * (303.461789035142376 + t * (5.7574879114754736 + t * 0.00275088986849762610))
                )
            )
            / (63493.915041308052 + t * (19070.1178243603945 + t * (1962.19362141235102 + t * (79.250704958640158 + t))))
        )
    elif x < 0.0:
        s = -0.5 * x
        t = 1.0 - s
        fd = (
            149.462587768865243
            + t
            * (
                22.8125889885050154
                + t
                * (
                    -0.629256395534285422
                    + t
                    * (
                        9.08120441515995244
                        + t
                        * (
                            3.35357478401835299
                            + t
                            * (
                                -0.473677696915555805
                                + t * (-0.467190913556185953 + t * (-0.0880610317272330793 - t * 0.00262208080491572673))
                            )
                        )
                    )
                )
            )
        ) / (
            269.94660938022644
            + s
            * (
                343.6419926336247
                + s
                * (
                    323.9049470901941
                    + s
                    * (
                        218.89170769294024
                        + s * (102.31331350098315 + s * (36.319337289702664 + s * (8.3317401231389461 + s)))
                    )
                )
            )
        )
    elif x < 2.0:
        t = 0.5 * x
        fd = (
            71652.717119215557
            + t
            * (
                134954.734070223743
                + t
                * (
                    153693.833350315645
                    + t
                    * (
                        123247.280745703400
                        + t
                        * (
                            72886.293647930726
                            + t
                            * (
                                32081.2499422362952
                                + t * (10210.9967337762918 + t * (2152.71110381320778 + t * 232.906588165205042))
                            )
                        )
                    )
                )
            )
        ) / (
            105667.839854298798
            + t
            * (
                31946.0752989314444
                + t
                * (
                    71158.788776422211
                    + t
                    * (
                        15650.8990138187414
                        + t
                        * (
                            13521.8033657783433
                            + t * (1646.98258283527892 + t * (618.90691969249409 + t * (-3.36319591755394735 + t)))
                        )
                    )
                )
            )
        )
    elif x < 5.0:
        t = 1 / 3 * (x - 2.0)
        fd = (
            23744.8706993314289
            + t
            * (
                68257.8589855623002
                + t
                * (
                    89327.4467683334597
                    + t
                    * (
                        62766.3415600442563
                        + t
                        * (
                            20093.6622609901994
                            + t * (-2213.89084119777949 + t * (-3901.66057267577389 - t * 948.642895944858861))
                        )
                    )
                )
            )
        ) / (
            9488.61972919565851
            + t
            * (
                12514.8125526953073
                + t
                * (
                    9903.44088207450946
                    + t
                    * (
                        2138.15420910334305
                        + t * (-528.394863730838233 + t * (-661.033633995449691 + t * (-51.4481470250962337 + t)))
                    )
                )
            )
        )
    elif x < 10.0:
        t = 0.2 * x - 1.0
        fd = (
            (
                311337.452661582536
                + t
                * (
                    1.11267074416648198e6
                    + t
                    * (
                        1.75638628895671735e6
                        + t
                        * (
                            1.59630855803772449e6
                            + t
                            * (
                                910818.935456183774
                                + t * (326492.733550701245 + t * (65507.2624972852908 + t * 4809.45649527286889))
                            )
                        )
                    )
                )
            )
            / (
                39721.6641625089685
                + t
                * (
                    86424.7529107662431
                    + t
                    * (
                        88163.7255252151780
                        + t
                        * (
                            50615.7363511157353
                            + t * (17334.9774805008209 + t * (2712.13170809042550 + t * (82.2205828354629102 - t)))
                        )
                    )
                )
            )
            * 0.999999999999999877
        )

    elif x < 20.0:
        t = 0.1 * x - 1.0

        fd = (
            7.26870063003059784e6
            + t
            * (
                2.79049734854776025e7
                + t
                * (
                    4.42791767759742390e7
                    + t
                    * (
                        3.63735017512363365e7
                        + t * (1.55766342463679795e7 + t * (2.97469357085299505e6 + t * 154516.447031598403))
                    )
                )
            )
        ) / (
            340542.544360209743
            + t
            * (
                805021.468647620047
                + t
                * (
                    759088.235455002605
                    + t
                    * (
                        304686.671371640343
                        + t * (39289.4061400542309 + t * (582.426138126398363 + t * (11.2728194581586028 - t)))
                    )
                )
            )
        )
    elif x < 40.0:
        t = 0.05 * x - 1.0
        fd = (
            4.81449797541963104e6
            + t
            * (
                1.85162850713127602e7
                + t
                * (
                    2.77630967522574435e7
                    + t
                    * (
                        2.03275937688070624e7
                        + t * (7.41578871589369361e6 + t * (1.21193113596189034e6 + t * 63211.9545144644852))
                    )
                )
            )
        ) / (
            80492.7765975237449
            + t
            * (
                189328.678152654840
                + t
                * (
                    151155.890651482570
                    + t * (48146.3242253837259 + t * (5407.08878394180588 + t * (112.195044410775577 - t)))
                )
            )
        )
    else:
        w = 1.0 / (x * x)
        s = 1.0 - 1600.0 * w
        fd = (
            x
            * sqrt(x)
            * 0.666666666666666667
            * (
                1.0
                + w
                * (8109.79390744477921 + s * (342.069867454704106 + s * 1.07141702293504595))
                / (6569.98472532829094 + s * (280.706465851683809 + s))
            )
        )

    return fd
