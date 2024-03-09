import pytest
from sarkas.utilities.io import alpha_to_int


@pytest.mark.parametrize(
    "argument, expected_outcome",
    [
        ("1", 1),
        ("0", 0),
        ("12345", 12345),
    ],
)
def test_alpha_to_int(argument, expected_outcome):
    actual_outcome = alpha_to_int(argument)
    assert actual_outcome == expected_outcome
