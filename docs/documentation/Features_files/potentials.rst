==========
Potentials
==========
Sarkas supports a variety of potentials both built-in and user defined. Currently the potential functions that are
implemented include

- :ref:`Coulomb <coulomb_pot>`
- :ref:`Yukawa <yukawa_pot>`
- :ref:`Exact Gradient-corrected Screened Yukawa <egs_pot>`
- :ref:`Quantum Statistical Potential <qsp_pot>`
- :ref:`Moliere <moliere_pot>`
- :ref:`Lennard Jones <lj_pot>`

You can read more about each of these potentials in the corresponding sections below.
All the equations will be given in cgs units, however, for easy conversion, we define the charge

.. math::
   \bar{e}^2 = \frac{e^2}{4\pi \varepsilon_0},

which when substituted in gives the equivalent mks formula.

Electron parameters and thermodynamic formulas are given in `here <../../theory/electron_properties.rst>`_.

.. _coulomb_pot:

Coulomb Potential
-----------------
Two charged particle with charge numbers :math:`Z_a` and :math:`Z_b` interact with each other via the Coulomb potential
given by

.. math::
   U_{ab}(r) = \frac{Z_{a}Z_b\bar{e}^2}{r}.

where :math:`r` is the distance between ions, :math:`e` is the elementary charge.

.. _yukawa_pot:

Yukawa Potential
----------------
The Yukawa potential, or screened Coulomb potential, is widely used in the plasma community to describe the interactions
of positively charged ions in a uniform background of electrons. The form of the Yukawa potential for two ions of charge
number :math:`Z_a` and :math:`Z_b` is given by

.. math::
   U_{ab}(r) = \frac{Z_{a} Z_b \bar{e}^2}{r}e^{- r /\lambda_{\textrm{TF}}}, \quad \kappa = \frac{a_{\textrm{ws}}}{\lambda_{\textrm{TF}} }

where :math:`\lambda_{\textrm{TF}}` is the Thomas-Fermi wavelength and :math:`\kappa` is the screening parameter.
In Sarkas :math:`\kappa` can be given as an input or it can be calculated from the
:ref:`Thomas-Fermi Wavelength <theory/electron_properties:thomas-fermi wavelength>` formula.

Notice that when :math:`\kappa = 0` we recover the Coulomb Potential.

.. _egs_pot:

Exact Gradient-corrected Screened Yukawa Potential
--------------------------------------------------
The Yukawa potential is derived on the assumption that the electron gas behaves as an ideal Fermi gas.
Improvements in this theory can be achieved by considering density gradients and exchange-correlation effects.
Stanton and Murillo :cite:`Stanton2015`, using a DFT formalism, derived an exact-gradient corrected ion pair potential
across a wide range of densities and temperatures.

The exact-gradient screened (EGS) potential introduces new parameters that can be easily calculated from initial inputs.
Density gradient corrections to the free energy functional lead to the first parameter, :math:`\nu`,

.. math::
   \nu = - \frac{3\lambda}{\pi^{3/2}}  \frac{4\pi \bar{e}^2 \beta }{\Lambda_{e}} \frac{d}{d\eta} \mathcal I_{-1/2}(\eta),

where :math:`\lambda` is a correction factor; :math:`\lambda = 1/9` for the true gradient corrected Thomas-Fermi model
and :math:`\lambda = 1` for the traditional von Weissaecker model, :math:`\mathcal I_{-1/2}[\eta_0]` is the
:ref:`Fermi Integral <theory/electron_properties:fermi integral>` of order :math:`-1/2`, and :math:`\Lambda_e` is the
:ref:`theory/electron_properties:de Broglie wavelength` of the electrons.

In the case :math:`\nu < 1` the EGS potential takes the form

.. math::
   U_{ab}(r) = \frac{Z_a Z_b \bar{e}^2 }{2r}\left [ ( 1+ \alpha ) e^{-r/\lambda_-} + ( 1 - \alpha) e^{-r/\lambda_+} \right ],

with

.. math::
   \lambda_\pm^2 = \frac{\nu \lambda_{\textrm{TF}}^2}{2b \pm 2b\sqrt{1 - \nu}}, \quad \alpha = \frac{b}{\sqrt{b - \nu}},

where the parameter :math:`b` arises from exchange-correlation contributions, see below.
On the other hand :math:`\nu > 1`, the pair potential has the form

.. math::
   U_{ab}(r) = \frac{Z_a Z_b \bar{e}^2}{r}\left [ \cos(r/\gamma_-) + \alpha' \sin(r/\gamma_-) \right ] e^{-r/\gamma_+}

with

.. math::
   \gamma_\pm^2 = \frac{\nu\lambda_{\textrm{TF}}^2}{\sqrt{\nu} \pm b}, \quad \alpha' = \frac{b}{\sqrt{\nu - b}}.

Neglect of exchange-correlational effects leads to :math:`b = 1` otherwise

.. math::
   b = 1 - \frac{2}{8} \frac{1}{k_{\textrm{F}}^2 \lambda_{\textrm{TF}}^2 }  \left [ h\left ( \Theta \right ) - 2 \Theta h'(\Theta) \right ]

where :math:`k_{\textrm{F}}` is the Fermi wavenumber and :math:`\Theta = (\beta E_{\textrm{F}})^{-1}` is the electron
:ref:`theory/electron_properties:Degeneracy Parameter` calculated from the :ref:`theory/electron_properties:Fermi Energy`.

.. math::
   h \left ( \Theta \right) = \frac{N(\Theta)}{D(\Theta)}\tanh \left( \Theta^{-1} \right ),

.. math::
   N(\Theta) = 1 + 2.8343\Theta^2 - 0.2151\Theta^3 + 5.2759\Theta^4,

.. math::
   D \left ( \Theta \right ) = 1 + 3.9431\Theta^2 + 7.9138\Theta^4.

.. _qsp_pot:

Quantum Statistical Potentials
------------------------------
An extensive review on Quantum Statistical Potentials is given in :cite:`Jones2007`. The following module uses that as
the main reference.

Quantum Statistical Potentials are defined by three terms

.. math::
    U(r) = U_{\textrm{pauli}}(r) + U_{\textrm{coul}}(r) + U_{\textrm{diff} }(r)

where

.. math::
    U_{\textrm{pauli}}(r) = - k_BT \ln \left [ 1 - \frac{1}{2} \exp \left ( - 2\pi r^2/ \Lambda^2 \right ) \right ]

is due to the Pauli exclusion principle and it accounts for spin-averaged effects,

.. math::
    U_{\textrm{coul}}(r) = \frac{Z_a Z_b \bar{e}^2}{r}

is the usual Coulomb interaction between two charged particles with charge numbers :math:`Z_a,Z_b`,
and :math:`U_{\textrm{diff}}(r)` is a diffraction term. There are two possibilities for
the diffraction term. The most common is the Deutsch potential

.. math::
    U_{\textrm{deutsch}}(r) = \frac{Z_a Z_b \bar{e}^2}{r} e^{ - 2\pi r/\Lambda_{ab}}.

The second most common form is the Kelbg potential

.. math::
    U_{\textrm{kelbg}}(r) = - \frac{Z_a Z_b \bar{e}^2}{r} \left [  e^{- 2 \pi r^2/\Lambda_{ab}^2 }
    - \sqrt{2} \pi \frac{r}{\Lambda_{ab}} \textrm{erfc} \left ( \sqrt{ 2\pi}  r/ \Lambda_{ab} \right )
    \right ]

In the above equations the screening length :math:`\Lambda_{ab}` is the thermal de Broglie wavelength
between the two charges defined as

.. math::
   \Lambda_{ab} = \sqrt{\frac{2\pi \hbar^2}{\mu_{ab} k_BT}}, \quad  \mu_{ab} = \frac{m_a m_b}{m_a + m_b}

Note that the de Broglie wavelength is defined differently in :cite:`Hansen1981` hence the factor of :math:`2\pi` in
the exponential.

The long range part of the potential is computed using the PPPM algorithm where only the
:math:`U_{\textrm{coul}}(r)` term is split into a short range and long range part.

The choice of this potential is due to its widespread use in the High Energy Density Physics community.

.. _moliere_pot:

Moliere Potential
-----------------
Moliere-type potentials have the form

.. math::
    \phi(r) =  \frac{Z_a Z_b \bar{e}^2}{r} \left [ \sum_{j}^{3} C_j e^{-b_j r} \right]

with the contraint

.. math::
    \sum_{j}^{3} C_j  = 1

more info can be found in :cite:`Wilson1977`

.. _lj_pot:

Lennard Jones
-------------
Sarkas support the general form of the multispecies Lennard Jones potential

.. math::
    U_{\mu\nu}(r) = k \epsilon_{\mu\nu} \left [ \left ( \frac{\sigma_{\mu\nu}}{r}\right )^m -
    \left ( \frac{\sigma_{\mu\nu}}{r}\right )^n \right ],

where

.. math::
    k = \frac{n}{m-n} \left ( \frac{n}{m} \right )^{\frac{m}{n-m}}.

In the case of multispecies liquids we use the `Lorentz-Berthelot <https://en.wikipedia.org/wiki/Combining_rules>`_
mixing rules

.. math::
    \epsilon_{12} = \sqrt{\epsilon_{11} \epsilon_{22}}, \quad \sigma_{12} = \frac{\sigma_{11} + \sigma_{22}}{2}.
