.. _force_error:

===========
Force Error
===========

The Force error is the error incurred when we cut the potential interaction after a certain distance. Following the works
of :cite:`Kolafa1992,Stern2008,Dharuman2017` we define the total force error for our P3M algorithm as

.. math::

    \Delta F_{\textrm{tot}} = \sqrt{ \Delta F_{\mathcal R}^2 + \Delta F_{\mathcal F}^2 }

where :math:`\Delta F_{\mathcal R}` is the error obtained in the PP part of the force calculation and
:math:`\Delta F_{\mathcal F}` is the error obtained in the PM part, the subscripts :math:`\mathcal{R, F}` stand for
real space and Fourier space respectively. :math:`\Delta F_{\mathcal R}` is calculated as follows

.. math::

    \Delta F_{\mathcal R} = \sqrt{\frac{N}{V} } \left [ \int_{r_c}^{\infty} d^3r
        \left | \nabla \phi_{\mathcal R}( \mathbf r) \right |^2  \right ]^{1/2},

where :math:`\phi_{\mathcal R}( \mathbf r)` is the short-range part of the chosen potential. In our example case of a
Yukawa potential we have

.. math::

    \phi_{\mathcal R}(r) = \frac{Q^2}{2r}
        \left [ e^{- \kappa r} \text{erfc} \left( \alpha r - \frac{\kappa}{2\alpha} \right )
            + e^{\kappa r} \text{erfc} \left( \alpha r + \frac{\kappa}{2\alpha} \right ) \right ],

where :math:`\kappa, \alpha` are the dimensionless screening parameter and Ewald parameter respectively and, for the
sake of clarity, we have a charge :math:`Q = Ze/\sqrt{4\pi \epsilon_0}` with an ionization state of :math:`Z = 1`. Integrating this potential,
and neglecting fast decaying terms, we find

.. math::

    \Delta F_{\mathcal R} \simeq 2 Q^2 \sqrt{\frac{N}{V}} \frac{e^{-\alpha^2 r_c^2}}{\sqrt{r_c}} e^{-\kappa^2/4 \alpha^2}.

On the other hand :math:`\Delta F_{\mathcal F}` is calculated from the following formulas

.. math::

    \Delta F_{\mathcal F} =  \sqrt{\frac{N}{V}} \frac{Q^2 \chi}{\sqrt{V^{1/3}}}

.. math::

    \chi^2V^{2/3}  = \left ( \sum_{\mathbf k \neq 0} G_{\mathbf k}^2 |\mathbf k |^2 \right )
        - \sum_{\mathbf n} \left [ \frac{\left ( \sum_{\mathbf m} \hat{U}_{\mathbf{k + m}}^2
        G_{\mathbf{k+m}} \mathbf{k_n} \cdot \mathbf{k_{n + m}} \right )^2 }{ \left( \sum_{\mathbf m} \hat{U}_{\mathbf{k_{n+m}}}^2 \right )^2 |\mathbf{k_{n} }|^2 } \right ].

This is a lot to take in, so let's unpack it. The first term is the RMS of the force field in Fourier space
obtained from solving Poisson's equation :math:`-\nabla \phi(\mathbf r) = \delta( \mathbf r - \mathbf r')` in Fourier
space. In a raw Ewald algorithm this term would be the PM part of the force. However, the P3M variant
solves Poisson's equation on a Mesh, hence, the second term which is non other than the RMS of the force obtained on the mesh.
:math:`G_{\mathbf k}` is the optimal Green's function which for the Yukawa potential is

.. math::
    G_{\mathbf k} = \frac{4\pi e^{-( \kappa^2 + \left |\mathbf k \right |^2)/(4\alpha^2)} }{\kappa^2 + |\mathbf {k}|^2}

where

.. math::

     \mathbf k ( n_x, n_y, n_z) = \mathbf{k_n} = \left ( \frac{2 \pi n_x}{L_x},
                                                        \frac{2 \pi n_y}{L_y},
                                                        \frac{2 \pi n_z}{L_z} \right ).

:math:`\hat{U}_{\mathbf k}` is the Fourier transform of the B-spline of order :math:`p`

.. math::

    \hat U_{\mathbf{k_n}} = \left[ \frac{\sin(\pi n_x /M_x) }{ \pi n_x/M_x} \right ]^p
    \left[ \frac{\sin(\pi n_y /M_y) }{ \pi n_y/M_y} \right ]^p
    \left[ \frac{\sin(\pi n_z /M_z) }{ \pi n_z/M_z} \right ]^p,

where :math:`M_{x,y,z}` is the number of mesh points along each direction. Finally the :math:`\mathbf{m}` refers to the
triplet of grid indices :math:`(m_x,m_y,m_z)` that contribute to aliasing. Note that in the above equations
as :math:`\kappa \rightarrow 0` (Coulomb limit), we recover the corresponding error estimate for the Coulomb potential.

The reason for this discussion is that by inverting the above equations we can find optimal parameters
:math:`r_c,\; \alpha` given some desired errors :math:`\Delta F_{\mathcal {R,F}}`. While
the equation for :math:`\Delta F_{\mathcal R}` can be easily inverted for :math:`r_c`, such task seems impossible for
:math:`\Delta F_{\mathcal F}` without having to calculate a Green's function for each chosen :math:`\alpha`. As you can
see in the second part of the output the time it takes to calculate :math:`G_{\mathbf k}` is in the order of seconds,
thus, a loop over several :math:`\alpha` values would be very time consuming. Fortunately researchers
have calculated an analytical approximation allowing for the exploration of the whole :math:`r_c,\; \alpha` parameter
space :cite:`Dharuman2017`. The equations of this approximation are

.. math::
    \Delta F_{\mathcal F}^{(\textrm{approx})} \simeq Q^2 \sqrt{\frac{N}{V}} A_{\mathcal F}^{1/2},

.. math::
    A_{\mathcal F} \simeq \frac{3}{2\pi^2} \sum_{m = 0}^{p -1 } C_{m}^{(p)} \left ( \frac{h}2 \right )^{2 (p + m)}
                            \frac{2}{1 + 2(p + m)} \beta(p,m),

.. math::
    \beta(p,m) = \int_0^{\infty} dk \; G_k^2 k^{2(p + m + 2)},

where :math:`h = L_x/M_x` and the coefficients :math:`C_m^{(p)}` are listed in Table I of :cite:`Deserno1998`.

Finally, by calculating

.. math::

    \Delta F_{\textrm{tot}}^{(\textrm{apprx})}( r_c, \alpha) = \sqrt{ \Delta F_{\mathcal R}^2 +
            ( \Delta F_{\mathcal F}^{(\textrm{approx})} ) ^2 }

we are able to investigate which parameters :math:`r_c,\; \alpha` are optimal for our simulation.

.. bibliography::
   :cited:
