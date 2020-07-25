.. _potentials:

==========
Potentials
==========
Sarkas supports a variety of potentials both built-in and user defined. Currently the potential functions that are implemented include

- Coulomb
- Yukawa
- Exact Gradient-corrected Screened Yukawa
- Quantum Statistical Potential
- Moliere 
- Lennard Jones

You can read more about each of these potentials in the corresponding sections below.
All the equations will be given in cgs units, however, for easy conversion, we define the charge  

.. math:: 
   \bar{e}^2 = \frac{e^2}{4\pi \varepsilon_0},

which when substituted in gives the equivalent mks formula. 

Coulomb Potential 
-----------------
Two charged particle with charge numbers :math:`Z_a` and :math:`Z_b` interact with each other via the Coulomb potential given by

.. math::
   V(r) = \frac{Z_{a}Z_b\bar{e}^2}{r}.

where :math:`r` is the distance between ions, :math:`e` is the elementary charge.


Yukawa Potential 
----------------
The Yukawa potential, or screened Coulomb potential, is widely used in the plasma community to describe the interactions of positively charged ions in a uniform background of electrons. The form of the Yukawa potential for two ions of charge number :math:`Z_a` and :math:`Z_b` is given by

.. math::
   V(r) = \frac{Z_{a} Z_b \bar{e}^2}{r}e^{-\kappa r},

where :math:`\kappa = 1/\lambda_{\text{TF}}` is the screening parameter. In Sarkas this can be either given as an input or it can be calculated from the Thomas-Fermi formula 

.. math::
   \frac{1}{\lambda_{TF}^2} = \frac{4 \sqrt{\pi} \bar{e}^2 \beta}{\Lambda_e^3}\frac{\mathcal I_{-1/2}(\eta)}{2}, 

where :math:`\Lambda_e` is the thermal de Broglie wavelength of the electrons and :math:`\mathcal I_p(\eta)` is the Dirac-Fermi integral of order :math:`p` defined as

.. math::
   \mathcal I_p ( \eta) = \int_0^\infty dx \frac{x^p}{1 + e^{x- \eta}}.

where :math:`\eta = \beta \mu` is the chemical potential. 
Notice that when :math:`\kappa = 0` we recover the Coulomb Potential.

Exact Gradient-corrected Screened Yukawa Potential
--------------------------------------------------
The Yukawa potential is derived on the assumption that the electron gas behaves as an ideal Fermi gas. Improvements in this theory can be achieved by considering density gradients and exchange-correlation effects. Stanton and Murillo [Stanton2015]_, using a DFT formalism derived an exact-gradient corrected ion pair potential across a wide range of densities and temperatures. The exact-gradient screened (EGS) potential introduces new parameters that can be easily calculated from initial inputs. Density gradient corrections to the free energy functional lead to the first parameter, :math:`\nu`,

.. math::
   \nu = - 3 \lambda \frac{4\pi \bar{e}^2 \beta }{ 4 \pi \Lambda_{e}} \mathcal I_{-3/2}(\eta),
  
where :math:`\lambda` is a correction factor; :math:`\lambda = 1/9` for the true gradient corrected Thomas-Fermi model and :math:`\lambda = 1` for the traditional  von Weissaecker model. In the case :math:`\nu < 1` the EGS potential takes the form

.. math::
   V(r) = \frac{Z_a Z_b \bar{e}^2 }{2r}\left [ ( 1+ \alpha ) e^{-r/\lambda_-} + ( 1 - \alpha) e^{-r/\lambda_+} \right ],
   
with 

.. math::
   \lambda_\pm^2 = \frac{\nu \lambda_{\textrm{TF}}^2}{2b \pm 2b\sqrt{1 - \nu}}, \quad \alpha = \frac{b}{\sqrt{b - \nu}},
   
where the parameter :math:`b` arises from exchange-correlation contributions, see below. 
On the other hand :math:`\nu > 1`, the pair potential has the form

.. math::
   V(r) = \frac{Z_a Z_b \bar{e}^2}{r}\left [ \cos(r/\gamma_-) + \alpha' \sin(r/\gamma_-) \right ] e^{-r/\gamma_+}
    
with 

.. math::
   \gamma_\pm^2 = \frac{\nu\lambda_{\textrm{TF}}^2}{\sqrt{\nu} \pm b}, \quad \alpha' = \frac{b}{\sqrt{\nu - b}}.

Neglect of exchange-correlational effects leads to :math:`b = 1` otherwise 

.. math::
   b = 1 - \frac{\Theta}8 \left [ h\left ( \Theta \right ) - 2 \Theta h'(\Theta) \right ]

where :math:`\Theta = (\beta E_F)^{-1}` and

.. math::
   h \left ( \Theta \right) = \frac{N(\Theta)}{D(\Theta)}\tanh \left( \Theta^{-1} \right ),

.. math::
   N(\Theta) = 1 + 2.8343\Theta^2 - 0.2151\Theta^3 + 5.2759\Theta^4,

.. math::
   D \left ( \Theta \right ) = 1 + 3.9431\Theta^2 + 7.9138\Theta^4.


Quantum Statistical Potential
-----------------------------
.. math::
   \phi(r) =  \frac{Z_a Z_b \bar{e}^2}{r} \left ( 1 - e^{ - 2\pi r/\Lambda_{ab}}\right ) + \delta_{ae} \delta_{be} k_BT \ln(2) \exp \left \{ - \frac{4 \pi r^2}{\Lambda_{ab}^2 \ln (2)} \right \}

where

.. math::
   \Lambda_{ab} = \sqrt{\frac{2\pi \hbar^2}{\mu_{ab} k_BT}},

and

.. math::
   \mu_{ab} = \frac{m_a m_b}{m_a + m_b}

is the thermal de Broglie wavelength between particles :math:`a` and :math:`b`. The last term, present only in the
interaction between two electrons, accounts for spin-averaged effects. The choice of this potential is due to its
widespread use in the High Energy Density Physics community.

