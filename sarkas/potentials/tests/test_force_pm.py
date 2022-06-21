from numpy import (
    arange,
    array,
    dtype,
    exp,
    imag,
    isclose,
    ndarray,
    pi,
    real,
    sin,
    sqrt,
    zeros,
    zeros_like,
)
from scipy.constants import epsilon_0

from ..force_pm import assgnmnt_func, create_k_arrays, force_optimized_green_function


def test_create_k_arrays():
    N = 1000
    box_lengths = (4.0 * pi * N / 3) ** (1.0 / 3.0) * array([1.0, 1.0, 1.0])
    mesh_sizes = array([2, 2, 2])

    kx, ky, kz = create_k_arrays(mesh_sizes, box_lengths)

    assert kx.shape == (1, mesh_sizes[0])

    assert ky.shape == (mesh_sizes[1], 1)

    assert kz.shape == (mesh_sizes[2], 1, 1)

    assert isinstance(kx, ndarray)
    assert isinstance(ky, ndarray)
    assert isinstance(kz, ndarray)

    assert kx.dtype == dtype("float64")
    assert ky.dtype == dtype("float64")
    assert kz.dtype == dtype("float64")

    # Check values
    kx_t = array([[-0.38977771, 0.0]])
    ky_t = array([[-0.38977771], [0.0]])
    kz_t = array([[[-0.38977771]], [[0.0]]])

    assert isclose(kx, kx_t).all()

    assert isclose(ky, ky_t).all()

    assert isclose(kz, kz_t).all()


def test_crete_k_arrays_2D():

    N = 1000
    box_lengths = sqrt(pi * N) * array([1.0, 1.0, 0.0])
    box_lengths[2] = 1.0
    mesh_sizes = array([2, 2, 1])

    kx, ky, kz = create_k_arrays(mesh_sizes, box_lengths)

    assert kx.shape == (1, mesh_sizes[0])

    assert ky.shape == (mesh_sizes[1], 1)

    assert kz.shape == (mesh_sizes[2], 1, 1)

    assert isinstance(kx, ndarray)
    assert isinstance(ky, ndarray)
    assert isinstance(kz, ndarray)

    assert kx.dtype == dtype("float64")
    assert ky.dtype == dtype("float64")
    assert kz.dtype == dtype("float64")

    # Check values
    kx_t = array([[-0.11209982, 0.0]])
    ky_t = array([[-0.11209982], [0.0]])
    kz_t = array([[[0.0]]])

    assert isclose(kx, kx_t).all()

    assert isclose(ky, ky_t).all()

    assert isclose(kz, kz_t).all()


def test_fogf():
    N = 1000
    box_lengths = (4.0 * pi * N / 3) ** (1.0 / 3.0) * array([1.0, 1.0, 1.0])
    kappa = 0.1
    alpha_ewald = 1.0
    mesh_sizes = array([2, 2, 2])
    aliases = array([3, 3, 3])
    cao = array([3, 3, 3])
    h_array = box_lengths / mesh_sizes
    G_k, kx_v, ky_v, kz_v, PM_err = force_optimized_green_function(
        box_lengths, h_array, mesh_sizes, aliases, cao, array([kappa, alpha_ewald, 1.0])
    )

    # Check dimensions
    assert (G_k.shape == array([mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]])).all()

    assert kx_v.shape == (1, mesh_sizes[0])

    assert ky_v.shape == (mesh_sizes[1], 1)

    assert kz_v.shape == (mesh_sizes[2], 1, 1)

    # Check values
    kx_t = array([[-0.38977771, 0.0]])
    ky_t = array([[-0.38977771], [0.0]])
    kz_t = array([[[-0.38977771]], [[0.0]]])

    G_k_t = array(
        [
            [[2.91525510e-03, 4.12033578e-04], [4.12033578e-04, 5.82591449e-05]],
            [[4.12033578e-04, 5.82591449e-05], [5.82591449e-05, 0.00000000e00]],
        ]
    )

    assert isclose(kx_v, kx_t).all()

    assert isclose(ky_v, ky_t).all()

    assert isclose(kz_v, kz_t).all()

    assert isclose(PM_err, 3.94467967822058)

    assert isclose(G_k, G_k_t).all()


def test_assignment_function():
    # Check that it returns the correct values
    cao = 3
    delta_x = 0.3
    wx = assgnmnt_func(cao, delta_x)

    assert isclose(wx, array([0.02, 0.66, 0.32])).all()
    # The sum of W(x) should be 1
    assert wx.sum() == 1

    # I should not get any negative numbers from W(x).
    # If I do it means that I am not choosing the closest point (or midpoint)
    delta_x = 0.89
    wx = assgnmnt_func(cao, delta_x)
    assert isclose(wx, array([0.07605, -0.0421, 0.96605])).all()
    # Note that the sum of the above is still 1.0.

    # Check that it returns the correct values
    cao = 4
    delta_x = 0.1
    wx = assgnmnt_func(cao, delta_x)

    assert isclose(wx, array([0.01066667, 0.41466667, 0.53866667, 0.036])).all()
    # The sum of W(x) should be 1
    assert wx.sum() == 1


def test_calc_charge_dens():
    pass
