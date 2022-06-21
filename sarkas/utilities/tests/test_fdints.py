"""Test module of fdints."""
from numpy import array, isclose, pi, sqrt

from ..fdints import (
    fd0h,
    fd1h,
    fd2h,
    fd3h,
    fd4h,
    fd5h,
    fd6h,
    fd7h,
    fd8h,
    fd9h,
    fd10h,
    fd11h,
    fd12h,
    fd13h,
    fd14h,
    fd15h,
    fd16h,
    fd17h,
    fd18h,
    fd19h,
    fd20h,
    fd21h,
    fdm1h,
    fdm3h,
    fdm5h,
    fdm7h,
    fdm9h,
    fermidirac_integral,
    invfd1h,
)

ETAS = array([-3.0, -1.0, 1.0, 3.5, 7.5, 15.0, 30.0, 60.0])
ORDERS = array(
    [
        -9 / 2,
        -7 / 2,
        -5 / 2,
        -3 / 2,
        -1 / 2,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0,
        6.5,
        7.0,
        7.5,
        8.0,
        8.5,
        9.0,
        9.5,
        10.0,
        10.5,
    ]
)


def test_invfd1h():
    """Test the calculation of eta."""

    # Values taken from Tutorial Notebooks
    ne = 1.62e32  # [N/m^3]
    lambda_deB = 1.957093e-11  # [m]
    u = lambda_deB**3 * ne * sqrt(pi) / 4.0
    eta = invfd1h(u)

    assert isclose(eta, -0.28604631569655325)


def test_fdm9h():

    results = array(
        [
            7.240015921650838e-03,
            -2.442145618002432e-02,
            1.647586588274429e-02,
            1.029804636049462e-03,
            -4.060975578007697e-04,
            -2.508232176525567e-05,
            -1.990373968491516e-06,
            -1.720097159754510e-07,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fdm9h(eta)
        assert isclose(fd, rs)


def test_fdm7h():

    results = array(
        [
            -3.545676388861523e-02,
            -2.657136470551858e-02,
            9.517508954351293e-02,
            -1.294450181808404e-02,
            -3.613853399968076e-03,
            -4.935123501719669e-04,
            -8.248542579808003e-05,
            -1.440219364000877e-05,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fdm7h(eta)
        assert isclose(fd, rs)


def test_fdm5h():

    results = array(
        [
            1.024984312340880e-01,
            3.492797177355019e-01,
            2.502420253443888e-02,
            -1.233961959341110e-01,
            -3.760442897358426e-02,
            -1.182183957963569e-02,
            -4.085597149537218e-03,
            -1.436908659724343e-03,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fdm5h(eta)
        assert isclose(fd, rs)


def test_fdm3h():

    results = array(
        [
            -1.647804107730068e-01,
            -8.394844152932343e-01,
            -1.580053773447138e00,
            -1.176280028545422e00,
            -7.505784586840675e-01,
            -5.193758746338315e-01,
            -3.656546825930124e-01,
            -2.582876225442183e-01,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fdm3h(eta)
        assert isclose(fd, rs)


def test_fdm1h():

    results = array(
        [
            8.525970123326887e-02,
            5.211503831079912e-01,
            1.820411357146962e00,
            3.591334824651103e00,
            5.432873628864860e00,
            7.731513057173728e00,
            1.094942130440661e01,
            1.549016158518217e01,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fdm1h(eta)
        assert isclose(fd, rs)


def test_fd0h():

    results = array(
        [
            4.858735157374205e-02,
            3.132616875182229e-01,
            1.313261687518223e00,
            3.529750418272620e00,
            7.500552931475361e00,
            1.500000030590227e01,
            3.000000000000009e01,
            6.000000000000000e01,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd0h(eta)
        assert isclose(fd, rs)


def test_fd1h():

    results = array(
        [
            4.336636755041557e-02,
            2.905008961699176e-01,
            1.396375280666564e00,
            4.837065897622567e00,
            1.399909743357599e01,
            3.894304660093270e01,
            1.096948183372665e02,
            3.099448732700438e02,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd1h(eta)
        assert isclose(fd, rs)


def test_fd2h():

    results = array(
        [
            4.918072033882423e-02,
            3.386479964034522e-01,
            1.806286070444774e00,
            7.739961645298564e00,
            2.976938105893487e01,
            1.141449337609459e02,
            4.516449340668481e02,
            1.801644934066848e03,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd2h(eta)
        assert isclose(fd, rs)


def test_fd3h():

    results = array(
        [
            6.561173880637544e-02,
            4.608488062901017e-01,
            2.661682624732004e00,
            1.365420168610915e01,
            6.833812856132876e01,
            3.581122477085265e02,
            1.985311377746038e03,
            1.117330291388515e04,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd3h(eta)
        assert isclose(fd, rs)


def test_fd4h():

    results = array(
        [
            9.896340290959225e-02,
            7.051297585956156e-01,
            4.328331225625401e00,
            2.586637394510408e01,
            1.653001170950006e02,
            1.174348022617251e03,
            9.098696044010894e03,
            7.219739208802179e04,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd4h(eta)
        assert isclose(fd, rs)


def test_fd5h():

    results = array(
        [
            1.647403937322051e-01,
            1.185968175443467e00,
            7.626535355005596e00,
            5.186981146923186e01,
            4.158852838077063e02,
            3.974487881941656e03,
            4.292925758509993e04,
            4.799485008429281e05,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd5h(eta)
        assert isclose(fd, rs)


def test_fd6h():

    results = array(
        [
            2.978018784709205e-01,
            2.159839661017986e00,
            1.438935649349364e01,
            1.091505015453568e02,
            1.079959324343085e03,
            1.377794488724111e04,
            2.069526863744442e05,
            3.257776652315915e06,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd6h(eta)
        assert isclose(fd, rs)


def test_fd7h():

    results = array(
        [
            5.778455375087482e-01,
            4.213264071926359e00,
            2.883131841599375e01,
            2.396666058159899e02,
            2.880213529254126e03,
            4.868423690758540e04,
            1.014417201742514e06,
            2.246912083856067e07,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd7h(eta)
        assert isclose(fd, rs)


def test_fd8h():

    results = array(
        [
            1.193042623617263e00,
            8.732138288905944e00,
            6.096945037216665e01,
            5.469755138110205e02,
            7.862865080220864e03,
            1.747634735470307e05,
            5.039016606494085e06,
            1.569439504883058e08,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd8h(eta)
        assert isclose(fd, rs)


def test_fd9h():

    results = array(
        [
            2.603141667351258e00,
            1.910506806522883e01,
            1.354192084601931e02,
            1.293868145709697e03,
            2.192168139830375e04,
            6.358400491901155e05,
            2.530631914465860e07,
            1.107558362743446e09,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd9h(eta)
        assert isclose(fd, rs)


def test_fd10h():

    results = array(
        [
            5.969820680488160e00,
            4.389948426765305e01,
            3.146680541843087e02,
            3.165640736730491e03,
            6.231539440840174e04,
            2.340617854292586e06,
            1.282644990485829e08,
            7.883001082246370e09,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd10h(eta)
        assert isclose(fd, rs)


def test_fd11h():

    results = array(
        [
            1.432510773316834e01,
            1.054873701421355e02,
            7.624071301487685e02,
            7.997774066508814e03,
            1.804014784983563e05,
            8.706355279385276e06,
            6.552423229390368e08,
            5.651215902520975e10,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd11h(eta)
        assert isclose(fd, rs)


def test_fd12h():

    results = array(
        [
            3.583278660538172e01,
            2.641275790125708e02,
            1.920621491104163e03,
            2.083671641045758e04,
            5.314330295476191e05,
            3.269159748061942e07,
            3.370296449774471e09,
            4.076323551443539e11,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd12h(eta)
        assert isclose(fd, rs)


def test_fd13h():

    results = array(
        [
            9.313870302507033e01,
            6.870205993243595e02,
            5.017993045210247e03,
            5.591696834104130e04,
            1.592092071043047e06,
            1.238219532435030e08,
            1.744018597946511e10,
            2.956078846984141e12,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd13h(eta)
        assert isclose(fd, rs)


def test_fd14h():

    results = array(
        [
            2.508781184723398e02,
            1.851484886980967e03,
            1.356711459868958e04,
            1.544073285162487e05,
            4.848712724195143e06,
            4.727830603631850e08,
            9.073325961350024e10,
            2.153759508773864e13,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd14h(eta)
        assert isclose(fd, rs)


def test_fd15h():

    results = array(
        [
            6.986360585012507e02,
            5.157783274159222e03,
            3.788356128997377e04,
            4.383205315095304e05,
            1.500759095117367e07,
            1.818973214398967e09,
            4.743320667852189e11,
            1.575715160529580e14,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd15h(eta)
        assert isclose(fd, rs)


def test_fd16h():

    results = array(
        [
            2.007219646720646e03,
            1.482234071460382e04,
            1.090540532961069e05,
            1.277985060309017e06,
            4.720135119144394e07,
            7.049084121468240e09,
            2.490622378495212e12,
            1.157079836302754e15,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd16h(eta)
        assert isclose(fd, rs)


def test_fd17h():

    results = array(
        [
            5.938814014650166e03,
            4.386311784955336e04,
            3.231138561347001e05,
            3.823781896168859e06,
            1.508420732678749e08,
            2.750798130249417e10,
            1.313056860664467e13,
            8.524941900028644e15,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd17h(eta)
        assert isclose(fd, rs)


def test_fd18h():

    results = array(
        [
            1.806585371781600e04,
            1.334484320310655e05,
            9.839000912153142e05,
            1.173074963158710e07,
            4.897833845586907e08,
            1.080715538012005e11,
            6.948254776893934e13,
            6.299767361155796e16,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd18h(eta)
        assert isclose(fd, rs)


def test_fd19h():

    results = array(
        [
            5.642067020436533e04,
            4.168044535669152e05,
            3.074986313749142e06,
            3.686910477586340e07,
            1.615856749616024e09,
            4.273879330919564e11,
            3.689532972232714e14,
            4.668155109009914e17,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd19h(eta)
        assert isclose(fd, rs)


def test_fd20h():

    results = array(
        [
            1.806629241770091e05,
            1.334722123421517e06,
            9.851381090333600e06,
            1.186176377953597e08,
            5.416646367586654e09,
            1.701154132363905e12,
            1.965505543248569e15,
            3.467790683237043e18,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd20h(eta)
        assert isclose(fd, rs)


def test_fd21h():

    results = array(
        [
            5.924272114982210e05,
            4.376998997938984e06,
            3.231634166275790e07,
            3.903364896044762e08,
            1.845016484905844e10,
            6.814687278457835e12,
            1.050272666569349e16,
            2.582022264434903e19,
        ]
    )

    for eta, rs in zip(ETAS, results):
        fd = fd21h(eta)
        assert isclose(fd, rs)


def test_fermidirac_integral():

    results = array(
        [
            7.240015921650838e-03,
            -3.545676388861523e-02,
            1.024984312340880e-01,
            -1.647804107730068e-01,
            8.525970123326887e-02,
            4.858735157374205e-02,
            4.336636755041557e-02,
            4.918072033882423e-02,
            6.561173880637544e-02,
            9.896340290959225e-02,
            1.647403937322051e-01,
            2.978018784709205e-01,
            5.778455375087482e-01,
            1.193042623617263e00,
            2.603141667351258e00,
            5.969820680488160e00,
            1.432510773316834e01,
            3.583278660538172e01,
            9.313870302507033e01,
            2.508781184723398e02,
            6.986360585012507e02,
            2.007219646720646e03,
            5.938814014650166e03,
            1.806585371781600e04,
            5.642067020436533e04,
            1.806629241770091e05,
            5.924272114982210e05,
        ]
    )

    for ord, res in zip(ORDERS, results):
        fd = fermidirac_integral(p=ord, eta=ETAS[0])
        isclose(fd, res)
