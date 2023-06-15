from numpy import array, isclose, pi
from scipy.constants import epsilon_0

from ..egs import egs_force

from pytest import mark

@mark.parametrize("test_params,expected_potential,expected_force",[
    ({"r":2.0,
      "alpha":1.3616,
      "lambda_p":1.778757e-09,
      "lambda_m":4.546000e-09,
      "charge":1.440961e-09},-0.9067719924627385,270184640.33105946)
    ],ids=["r-2;alpha-1.3616"])
def test_egs_force(test_params,expected_potential,expected_force):
    """Test the calculation of the bare egs force."""
    r = test_params["r"]
    alpha = test_params["alpha"]
    lambda_p = test_params["lambda_p"]
    lambda_m = test_params["lambda_m"]
    charge = test_params["charge"]
    c_const = charge**2 / (4.0 * pi * epsilon_0)
    pot_mat = array([c_const * 0.5, 1.0 + alpha, 1.0 - alpha, 1.0 / lambda_m, 1.0 / lambda_p, 1.0e-14])

    potential, force = egs_force(r, pot_mat)

    assert isclose(potential, expected_potential)

    assert isclose(force, expected_force)


# def test_update_params():
#     # TODO: write a test for update_params
#     pass
