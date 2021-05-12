===================
Electron Properties
===================

Below we show the equations used for the calculation of electron gas properties.

Fermi Integral
--------------
The Fermi integral :math:`\mathcal F_{p}`of order :math:`p` is defined as

.. math::

    \mathcal F_p [\eta] = \frac{1}{\Gamma( p +  1) } \mathcal I_{p} [\eta] = \frac{1}{\Gamma(p + 1) } \int_0^{\infty} dx \frac{x^p}{1 + e^{x - \eta} },

.. math::

    \Gamma (s) = \int_0^{\infty} dx x^{s - 1} e^{-x} dx.

The Fermi-Dirac integral satisfies the relation

.. math::

    \frac{d}{dx} \mathcal F_{p}(x) = \mathcal F_{p - 1}(x),

and here are some useful values of :math:`\Gamma(x)`

.. math::

    \Gamma \left(- \frac{5}{2} \right )  = -\frac{8\sqrt{\pi}}{15}, \quad \Gamma \left( \frac {5}{2} \right ) = \frac{3\sqrt{\pi} }{4},

.. math::

    \Gamma \left( - \frac{3}{2} \right ) = \frac{4\sqrt{\pi}}{3}, \quad \Gamma \left ( \frac {3}{2} \right ) = \frac{ \sqrt{\pi} }{2},

.. math::

    \Gamma \left (- \frac{1}{2} \right ) = - 2 \sqrt{\pi}, \quad \Gamma \left ( \frac {1}{2} \right ) = \sqrt{\pi}.


Thermodynamics Electron Gas
---------------------------
For future reference we define some of thermodynamics quantities of the unpolarized paramagnetic electron gas with dimensionless
chemical potential :math:`\eta = \beta \mu` and spin degeneracy :math:`g = 2`.

de Broglie wavelength
^^^^^^^^^^^^^^^^^^^^^
The de Broglie wavelength is given by

.. math::

   \Lambda_e = \sqrt{\frac{ 2\pi \hbar^2 \beta}{m_e}}.

Grand Potential
^^^^^^^^^^^^^^^
The grand thermodynamic potential :math:`\Omega` is given by

.. math::

    \Omega = - \frac{g \beta^{-1} }{\Lambda^3} \int d^3r \mathcal F_{3/2}[\eta] = - \frac 43 \frac{g \beta^{-1} }{\sqrt{\pi} \Lambda^3} \int d^3r \mathcal I_{3/2}\left [ \eta \right ]



Number of Particles
^^^^^^^^^^^^^^^^^^^
The total number of electrons :math:`N_e` is

.. math::

    N_e  = \frac{g}{\Lambda^3} \int d^3r \mathcal F_{1/2}[\eta]  =  \frac{g}{\Lambda^3} \frac{2}{\sqrt{\pi} } \int d^3r \mathcal I_{1/2}[\eta]


Pressure
^^^^^^^^
The pressure is given by the thermodynamic formula :math:`PV = - \Omega`

.. math::

    PV =  \frac{g\beta^{-1}}{\Lambda^3} \int d^3r \mathcal F_{3/2}[\eta] = \frac{4}{3 \sqrt{\pi} } \frac{g\beta^{-1} }{\Lambda^3} \int d^3r\mathcal I_{3/2}[\eta]

Internal Energy
^^^^^^^^^^^^^^^
The internal energy is given by

.. math::

    E = \frac{3}{2}  \frac{g \beta^{-1} }{\Lambda^3} \int d^3r \mathcal F_{3/2}[\eta] = \frac{2}{\sqrt{\pi} } \frac{g}{\beta\Lambda^3} \int d^3r \mathcal I_{3/2}[\eta]


which is equal to :math:`E = 3/2 PV`.

Free Energy
^^^^^^^^^^^
The Free energy is

.. math::

    F = \frac{g }{\beta \Lambda^3} \int d^3r \left ( \eta \mathcal F_{1/2}[\eta] -  \mathcal F_{3/2}[\eta] \right )  =
     \frac{2g}{ \beta \sqrt{\pi} \Lambda^3} \int d^3r \left ( \eta \mathcal I_{1/2}[\eta] - \frac 23 \mathcal I_{3/2}[\eta] \right ).

Notice that :math:`F` is a functional of the density :math:`N/V` not of :math:`n(\mathbf{r})`, because that is integrated out.

Entropy
^^^^^^^
The entropy is

.. math::

    S = \frac{g}{\Lambda^3} \int d^3r \left (\frac 52 \mathcal F_{3/2}[\eta] - \eta \mathcal F_{1/2}[\eta] \right ) = \frac{2 g}{\sqrt{\pi} \Lambda^3 } \int d^3r \left ( \frac 53 \mathcal I_{3/2}[\eta] - \eta \mathcal I_{1/2}[\eta] \right ).

Dimensionless Parameters
------------------------

Coupling parameters
^^^^^^^^^^^^^^^^^^^
.. math::

    \Gamma_e = \frac{\bar{e}^2\beta}{a_e}, \qquad  r_s = a_e/a_0

where :math:`a_e` is the Wigner-Seitz radius of the electron gas and :math:`a_0 = \hbar^2/m_e \bar{e}^2` is the Bohr radius.


Fermi Energy
^^^^^^^^^^^^
The Fermi energy of a non-interacting electron gas is calculated from the Fermi wave number :math:`k_F = (3 \pi^2 n)^{1/3}`

.. math::

    E_{\textrm F} = \frac{\hbar^2k_F^2}{2m_e} = \frac{\Lambda^2}{\beta} \left( \frac{3\sqrt{\pi}}{8} n \right )^{2/3}.

Degeneracy Parameter
^^^^^^^^^^^^^^^^^^^^
The above equation leads immediately to the degeneracy parameter

.. math::

    \Theta = \frac{k_BT}{E_{\textrm{F}}} = \left( \frac{3 \sqrt{\pi}}{8} \frac{n}{\Lambda^3} \right )^{-2/3} = \left( \frac 32 \mathcal I_{1/2}[\eta] \right )^{-2/3}.


Relativistic Parameter
^^^^^^^^^^^^^^^^^^^^^^
Relativistic effect are given by

.. math::

    x_F = \frac{\hbar k_F}{ m_e c}

Warm Dense Matter Parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::
    \mathcal W(n_e, \beta) = \mathcal S(\Gamma_e) \mathcal S(\Theta)

where :math:`\mathcal S(x) = 2/(1/x + x)`.

Landau Length
^^^^^^^^^^^^^
This is given by

.. math::

    l_{\textrm{L}} = 4\pi \bar{e}^2 \beta.

Thomas-Fermi Wavelength
^^^^^^^^^^^^^^^^^^^^^^^

This is given by

.. math::

    \lambda_{\textrm{TF}}^2 = \left ( \frac{ l_{\textrm{L}} }{\Lambda^3} g \mathcal F_{-1/2}[\eta] \right )^{-1} = \frac{\Lambda^3}{l_{\textrm{L}}} \left(  \frac{g}{\sqrt{\pi} } \mathcal I_{-1/2}[\eta] \right )^{-1}.
