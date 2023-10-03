from numpy import cos, exp, sin, sqrt


def const_exp(t, A, B):
    return A - exp(-t / B)


def exponential(t, A):
    return exp(-t / A)


def gaussian(t, a):
    return exp(-0.5 * t**2 / a**2)


def cf_second(t, w1, w2, C):
    """Second-order approx for continued fraction"""

    # w_bar = sqrt(w1**2 - 0.25 * w2**2)
    f = cos(w1 * t) + C * sin(w1 * t)
    return f * exp(-0.5 * w2 * t)


def cf_third(t, t1, t2, w1, w2, C):
    """Third-order approx for continued fraction"""

    return exp(-t / t1) * (C * exp(-t / t2) + sin(w1 * t) + (1 - C) * cos(w2 * t))


def rse(t, tau0, tauE, beta):

    argument = (tau0 / tauE) ** beta * (1 - (1 + (t / tau0) ** 2) ** (0.5 * beta))
    return exp(argument)


def acf_fit_p5(t, a1, t1, t3, b1, b3, w1):

    one = (1 - a1) * cos(w1 * t) * exp(-((t / t1) ** b1))
    three = a1 * exp(-((t / t3) ** b3))
    return one + three


def acf_fit_p11(t, a1, a2, t1, t2, t3, b1, b2, b3, w1, w2, p1, p2):

    one = (0.5 - a1) * cos(w1 * t + p1) * exp(-((t / t1) ** b1))
    two = (0.5 - a2) * cos(w2 * t + p2) * exp(-((t / t2) ** b2))
    three = (a1 + a2) * exp(-((t / t3) ** b3))
    return one + two + three
