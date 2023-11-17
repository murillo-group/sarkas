r"""
Module for handling custom potential of the form given below.

Potential Attributes
********************
The elements of the :attr:`sarkas.potentials.core.Potential.matrix` are:

.. code-block:: python

    pot_matrix[0] = q_iq_je^2/(4 pi eps_0) Force factor between two particles.
    pot_matrix[1] = kappa
    pot_matrix[2] = a
    pot_matrix[3] = b
    pot_matrix[4] = c
    pot_matrix[5] = d
    pot_matrix[6] = a_rs. Short-range cutoff.

"""
from numba import jit
from numba.core.types import float64, UniTuple
from numpy import inf, interp, array, loadtxt, pi, argmin, sqrt, where, zeros, interp, finfo, dtype
from scipy.integrate import quad


# @jit(UniTuple(float64, 2)(float64, float64[:, :]), nopython=True)
@jit(nopython = True)
def tab_force_nn(r, pot_matrix):
    """
    Numba'd function to calculate the PP force between particles using the Moliere Potential.

    Parameters
    ----------
    r : float
        Particles' distance.

    pot_matrix : numpy.ndarray
        Slice of `sarkas.potentials.Potential.matrix` containing the potential parameters.

    Returns
    -------
    u_r : float
        Potential.

    f_r : float
        Force between two particles.

    """

    # Unpack the parameters
    r_tab = pot_matrix[0, :]
    u_tab = pot_matrix[1, :]
    f_tab = pot_matrix[2, :]
    # dr = abs(r_tab[1] - r_tab[0])
    if r <= r_tab[-1]:
        dr = abs(r_tab[1] - r_tab[0])
        if r < r_tab[0]:
            rb = 0
        else:
            rb = int(r / dr)  - int(r_tab[0]/dr) * (r >= r_tab[0])

        # The following branchless programming is needed because numba grabs the wrong element when rbin > rc.
        u_r = (u_tab[rb] - 0.0 * (r - r_tab[-1]) * f_tab[-1] - 0.0*u_tab[-1]) * (rb < pot_matrix.shape[1]) + 0.0
        f_r = (f_tab[rb] - 0.0*f_tab[-1]) * (rb < pot_matrix.shape[1]) + 0.0
    else:
        # rb = 0
        u_r = 0.0
        f_r = 0.0

    return u_r, f_r

# @jit(UniTuple(float64, 2)(float64, float64[:, :]), nopython=True)
@jit(nopython = True)
def tab_force_lin_interp(r, pot_matrix):
    """
    Numba'd function to calculate the PP force between particles using the Moliere Potential.

    Parameters
    ----------
    r : float
        Particles' distance.

    pot_matrix : numpy.ndarray
        Slice of `sarkas.potentials.Potential.matrix` containing the potential parameters.

    Returns
    -------
    u_r : float
        Potential.

    f_r : float
        Force between two particles.

    """

        # Unpack the parameters
    r_tab = pot_matrix[0, :]
    u_tab = pot_matrix[1, :]
    f_tab = pot_matrix[2, :]

    if r <= r_tab[-1]:
        dr = abs(r_tab[1] - r_tab[0])
        if r < r_tab[0]:
            rbin = 0
        else:
            rbin = int(r / dr)  - int(r_tab[0]/dr) * (r >= r_tab[0])
        # If rbin is bigger than 0, the left bin is bin - 1 otherwise is 0
        bin0 = (rbin - 1) * (rbin > 0) + 0
        # The right bin is rbin + 1 if rbin is less then the max length - 1 
        bin1 = rbin + 1*( rbin < len(r_tab) - 2)

        r0 = r_tab[bin0]
        r1 = r_tab[bin1]

        u0 = u_tab[bin0]
        u1 = u_tab[bin1]

        f0 = f_tab[bin0]
        f1 = f_tab[bin1]

        u_r = u0 + (r - r0) * ( u1 - u0)/(r1 - r0)

        f_r = f0 + (r - r0) * ( f1 - f0)/(r1 - r0)
    else:
        u_r = 0.0
        f_r = 0.0
        # rbin = 0
        # bin0, bin1 = 0,0
    return u_r, f_r



def potential_derivatives(r, pot_matrix):
    """Calculate the first and second derivative of the potential.

    Parameters
    ----------
    r_in : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables.

    Returns
    -------
    u_r : float, numpy.ndarray
        Potential value.

    dv_dr : float, numpy.ndarray
        First derivative of the potential.

    d2v_dr2 : float, numpy.ndarray
        Second derivative of the potential.

    """

    # Unpack the parameters
    r_tab = pot_matrix[0, :]
    u_tab = pot_matrix[1, :]
    f_tab = pot_matrix[2, :]
    f2_tab = pot_matrix[3, :]

    u_r = interp(r, r_tab, u_tab)
    dv_dr = interp(r, r_tab, -f_tab)
    d2v_dr2 = interp(r, r_tab, f2_tab)

    return u_r, dv_dr, d2v_dr2


def pretty_print_info(potential):
    """
    Print potential specific parameters in a user-friendly way.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.

    """
    msg = (
        f"screening type : {potential.screening_length_type}\n"
        f"screening length = {potential.screening_length:.6e} {potential.units_dict['length']}\n"
        f"kappa = {potential.a_ws / potential.screening_length:.4f}\n"
        f"Gamma_eff = {potential.coupling_constant:.2f}\n"
    )
    print(msg)


def update_params(potential):
    """
    Assign potential dependent simulation's parameters.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.

    """
    r, u, f, f2 = loadtxt(potential.tabulated_file, skiprows=1, unpack=True, delimiter=",")
    dr = r[1] - r[2]
    # Select all the indices for which r is less than the cutoff radius
    mask = where(r <= potential.rc)[0]
    params_len = len(r[mask])

    potential.matrix = zeros((potential.num_species, potential.num_species, 4, params_len))
    beta = 1.0 / (potential.kB * potential.electron_temperature)
    
    for i, _ in enumerate(potential.species_charges):
        for j, _ in enumerate(potential.species_charges):
            # Increase the r array by epsilon so that you are certain to get the right bin later in tab_force
            potential.matrix[i, j, 0, :] = r[mask]
            potential.matrix[i, j, 1, :] = u[mask]
            potential.matrix[i, j, 2, :] = f[mask]
            potential.matrix[i, j, 3, :] = f2[mask]
            
        
    if hasattr(potential,"interpolation_type"):
        if potential.interpolation_type in ["linear", "lin"]:
           potential.force = tab_force_lin_interp
    else:
        potential.force = tab_force_nn
    
    potential.potential_derivatives = potential_derivatives
    potential.force_error = calc_force_error_quad(potential.a_ws, beta, potential.rc, potential.matrix[0, 0])


def force_error_integrand(r, pot_matrix):
    r"""Auxiliary function to be used in `scipy.integrate.quad` to calculate the integrand.

    Parameters
    ----------
    r_in : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        Slice of the `sarkas.potentials.core.Potential.matrix` containing the necessary potential parameters.

    Returns
    -------
    _ : float
        Integrand :math:`4\pi r^2 ( d r\phi(r)/dr )^2`

    """

    _, dv_dr, _ = potential_derivatives(r, pot_matrix)

    return 4.0 * pi * r**2 * dv_dr**2


def calc_force_error_quad(a, beta, rc, pot_matrix):
    r"""
    Calculate the force error by integrating the square modulus of the force over the neglected volume.\n
    The force error is calculated from

    .. math::
        \Delta F =  \left [ 4 \pi \int_{r_c}^{\infty} dr \, r^2  \left ( \frac{d\phi(r)}{r} \right )^2 ]^{1/2}

    where :math:`\phi(r)` is only the radial part of the potential, :math:`r_c` is the cutoff radius, and :math:`r` is scaled by the input parameter `a`.\n
    The integral is calculated using `scipy.integrate.quad`. The derivative of the potential is obtained from :meth:`potential_derivatives`.

    Parameters
    ----------
    a : float
        Rescaling length. Usually it is the Wigner-Seitz radius.

    rc : float
        Cutoff radius to be used as the lower limit of the integral. The lower limit is actually `rc /a`.

    pot_matrix: numpy.ndarray
        Slice of the `sarkas.potentials.Potential.matrix` containing the parameters of the potential. It must be a 1D-array.

    Returns
    -------
    f_err: float
        Force error. It is the sqrt root of the integral. It is calculated using `scipy.integrate.quad`  and :func:`potential_derivatives`.

    """

    params = pot_matrix.copy()
    params[0, :] /= a  # a
    params[1, :] *= beta  # a
    params[2, :] *= beta  # a

    r_c = rc / a

    result, _ = quad(force_error_integrand, a=r_c, b=inf, args=(params,))

    f_err = sqrt(result)

    return f_err
