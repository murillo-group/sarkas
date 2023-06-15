from numpy import (
    arange,
    array,
    dtype,
    exp,
    imag,
    isclose,
    meshgrid,
    mod,
    ndarray,
    pi,
    real,
    sin,
    sqrt,
    zeros,
    zeros_like,
)
from numpy.random import default_rng
from scipy.constants import epsilon_0

from ..force_pp import create_cells_array, create_head_list_arrays


def create_hexagonal_lattice(Nx, Ny, perturb):
    rng = default_rng(123456789)

    box_lengths = sqrt(pi) * array([Nx, Ny, 0.0])

    dx_lattice = box_lengths[0] / Nx  # Lattice spacing
    dy_lattice = box_lengths[1] / Ny  # Lattice spacing

    # Create x, y, and z position arrays
    if Ny > Nx:
        x = arange(0, box_lengths[0], dx_lattice) + 0.5 * dx_lattice
        y = arange(0, box_lengths[1], dy_lattice)

        # Create a lattice with appropriate x, y, and z values based on arange
        X, Y = meshgrid(x, y)
        # Shift the Y axis of every other row of particles
        Y[:, ::2] += dy_lattice / 2

    else:
        x = arange(0, box_lengths[0], dx_lattice)
        y = arange(0, box_lengths[1], dy_lattice) + 0.5 * dy_lattice

        # Create a lattice with appropriate x, y, and z values based on arange
        X, Y = meshgrid(x, y)
        # Shift the Y axis of every other row of particles
        X[::2, :] += dx_lattice / 2

    # Perturb lattice
    X += rng.uniform(-0.5, 0.5, X.shape) * perturb * dx_lattice
    Y += rng.uniform(-0.5, 0.5, Y.shape) * perturb * dy_lattice

    pos = zeros((Nx * Ny, 3))
    # Flatten the meshgrid values for plotting and computation
    pos[:, 0] = X.ravel()
    pos[:, 1] = Y.ravel()
    pos[:, 2] = 0.0

    return pos, box_lengths


def test_create_cells_array():
    box_lengths = array([10.0, 10.0, 10.0])
    cutoff = 4.0

    cells, cell_lengths = create_cells_array(box_lengths, cutoff)

    assert isinstance(cells, ndarray)
    assert isinstance(cell_lengths, ndarray)

    assert cells.dtype == dtype("int64")
    assert cell_lengths.dtype == dtype("float64")

    assert isclose(cells, 2).all()
    assert isclose(cell_lengths, 5.0).all()

def test_create_cells_array_2d():
    box_lengths = array([10.0, 10.0, 0.0])
    cutoff = 2.0

    cells, cell_lengths = create_cells_array(box_lengths, cutoff)

    assert isclose(cells[:2], 5).all()
    assert isclose(cell_lengths[:2], 2.0).all()
    # The third dimension should still be one cell with length 1.0. This is to avoid division by zero.
    assert isclose(cells[2], 0)
    assert isclose(cell_lengths[2], 0.0)

def test_create_cells_array_1d():
    box_lengths = array([20.0, 0.0, 0.0])
    cutoff = 2.5

    cells, cell_lengths = create_cells_array(box_lengths, cutoff)

    # The third dimension should still be one cell with length 1.0. This is to avoid division by zero.
    assert isclose(cells[0], 8).all()
    assert isclose(cell_lengths[0], 2.5).all()
    assert isclose(cells[1], 0)
    assert isclose(cell_lengths[1], 0.0)
    assert isclose(cells[2], 0)
    assert isclose(cell_lengths[2], 0.0)

def test_create_cells_array_hex_2d():
    pos, box_lengths = create_hexagonal_lattice(4, 5, 0.1)

    cutoff = box_lengths[0] / 3
    cells, cell_lengths = create_cells_array(box_lengths, cutoff)

    assert isclose(cells, array([3, 3, 0])).all()

    assert isclose(cell_lengths, array([2.3632718, 2.95408975, 0.0])).all()


def test_create_head_list_arrays_2d():
    N = 10
    box_lengths = sqrt(pi * N) * array([1.0, 1.0, 0.0])

    cutoff = box_lengths[0] / 3
    cells, cell_lengths = create_cells_array(box_lengths, cutoff)

    rng = default_rng(123456789)
    pos = rng.uniform(low=0.0, high=box_lengths[0], size=(N, 3))
    pos[:, 2] = 0.0

    head_true = array([5, 8, 6, -50, -50, 3, 0, 9, 7])
    ls_array_true = array([-50, -50, -50, 2, -50, 4, -50, -50, -50, 1])
    head, ls_array = create_head_list_arrays(pos, cell_lengths, cells)

    assert isinstance(head, ndarray)
    assert isinstance(ls_array, ndarray)

    assert head.dtype == dtype("int64")
    assert ls_array.dtype == dtype("int64")

    assert (head == head_true).all()
    assert (ls_array == ls_array_true).all()

def test_create_head_list_arrays_3d():
    N = 10
    box_lengths = (4.0 * pi * N / 3.0) ** (1 / 3.0) * array([1.0, 1.0, 1.0])

    cutoff = box_lengths[0] / 3
    cells, cell_lengths = create_cells_array(box_lengths, cutoff)

    rng = default_rng(123456789)
    pos = rng.uniform(low=0.0, high=box_lengths[0], size=(N, 3))

    head_true = array([5, 8, 6, -50, -50, 3, 0, 9, 7])
    ls_array_true = array([-50, -50, -50, 2, -50, 4, -50, -50, -50, 1])
    head, ls_array = create_head_list_arrays(pos, cell_lengths, cells)

    assert isinstance(head, ndarray)
    assert isinstance(ls_array, ndarray)

    assert head.dtype == dtype("int64")
    assert ls_array.dtype == dtype("int64")

    assert (head == head_true).all()
    assert (ls_array == ls_array_true).all()

def test_create_head_list_arrays_hex_2d():
    pos, box_lengths = create_hexagonal_lattice(4, 5, 0.1)

    cutoff = box_lengths[0] / 3
    cells, cell_lengths = create_cells_array(box_lengths, cutoff)

    head, ls_array = create_head_list_arrays(pos, cell_lengths, cells)

    assert isclose(head, array([4, 6, 7, 8, 13, 15, 16, 18, 19])).all()

    assert isclose(ls_array, array([-50, -50, 1, -50, 0, 2, 5, 3, -50, -50, 9, -50, -50, 10, -50, 11, 12, 14, 17, -50])).all()
