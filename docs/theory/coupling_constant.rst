=================
Coupling Constant
=================

The coupling constant is denoted by :math:`\Gamma` and it corresponds to

.. math::

    \Gamma = \frac{(Ze)^2}{4\pi \varepsilon_0 k_B T a_{\textrm{ws}}},

In the case of mixtures the :meth:`sarkas.core.Parameters.calc_coupling_constant` method calculates the effective :math:`\Gamma` as

.. math::

    \Gamma_{\textrm{eff}} = \sum_{\alpha} c_{\alpha} \Gamma_{\alpha}, \quad
                \Gamma_{\alpha} = \frac{(Z_{\alpha}e)^2}{4 \pi \varepsilon_0} \frac{1}{ k_B T a_{\alpha}},

where :math:`a_{\alpha}` is Wigner-Seitz radius of species :math:`\alpha` defined as

.. math::

    a_{\alpha} = a_{\textrm{ws}} \left ( \frac{Z_{\alpha}}{\left \langle Z \right \rangle} \right )^{1/3},

Note that we can rewrite the effective :math:`\Gamma` as

.. math::

    \Gamma_{\textrm{eff}} = \left \langle Z^{5/3} \right \rangle \left \langle Z \right \rangle^{1/3} \frac{e^2}{4\pi \varepsilon_0} \frac{1}{k_B T a_{\textrm{ws} }},

which is the definition usually found in the literature.