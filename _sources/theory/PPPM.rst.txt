===========================================
Particle-Particle Particle-Mesh Algorithm
===========================================

Ewald Sum
=========

Long range forces are calculated using the Ewald method which consists in dividing the potential into a short
and a long range part. Physically this is equivalent to adding and subtracting a screening cloud around each charge.
This screening cloud is usually chosen to be given by a Gaussian charged density distribution, but it need not be.
The choice of a Gaussian is due to spherical symmetry.

The total charge density at point :math:`\mathbf r` is then

.. math::

    \rho(\mathbf r) = \sum_{i}^N  \left \{ \left ( q_i\delta( \mathbf r - \mathbf r_i) - \frac{q_i\alpha^{3/2}}{\pi} e^{-\alpha^2 \left( \mathbf r - \mathbf r_i \right )^2 } \right ) + \frac{q_i\alpha^{3/2}}{\pi} e^{-\alpha^2 \left( \mathbf r- \mathbf r_i \right )^2 } \right \},

where the first term is the charge density due to the real particles and the last two terms are a negative
and positive screening cloud. The first two term are in parenthesis to emphasizes the splitting into

.. math::

    \rho(\mathbf r)  = \rho_{\mathcal R}(\mathbf r) + \rho_{\mathcal F}(\mathbf r)

.. math::

    \rho_{\mathcal R} (\mathbf r) = \sum_{i}^N \left ( q_i\delta( \mathbf r- \mathbf r_i) - \frac{q_i\alpha^{3/2}}{\pi} e^{-\alpha^2 \left( \mathbf r- \mathbf r_i \right )^2 } \right ), \quad \rho_{\mathcal F}(\mathbf r) = \sum_{i}^N \frac{q_i\alpha^{3/2}}{\pi} e^{-\alpha^2 \left( \mathbf r- \mathbf r_i \right )^2 }


where :math:`\rho_{\mathcal R}(\mathbf r)` indicates the charge density leading to the short range part of the potential
and :math:`\rho_{\mathcal F}(\mathbf r)` leading to the long range part.
The subscripts :math:`\mathcal R, \mathcal F` stand for Real and Fourier space indicating the way the calculation
will be done.

The potential at every point :math:`\mathbf r` is calculated from Poisson's equation

.. math::

    -\nabla^2 \phi( \mathbf r) = 4\pi \rho_{\mathcal R} (\mathbf r) + 4\pi \rho_{\mathcal F}( \mathbf r).

Short-range term
----------------

The short range term is calculated in the usual way

.. math::

    -\nabla^2 \phi_{\mathcal R}( \mathbf r) = 4\pi \sum_{i}^N  \left ( q_i\delta( \mathbf r- \mathbf r_i) - \frac{q_i\alpha^{3/2}}{\pi} e^{-\alpha^2 \left( \mathbf r- \mathbf r_i \right )^2 } \right ).

The first term :math:`\delta(\mathbf r - \mathbf r_i)` leads to the usual Coulomb potential (:math:`\sim 1/r`) while
the Gaussian leads to the error function

.. math::

    \phi_{\mathcal R}( \mathbf r ) = \sum_i^N  \frac{q_i}{r} - \frac{q_i}{r}\text{erf} (\alpha r)  = \sum_i^N \frac{q_i}{r} \text{erfc}(\alpha r)

Long-range term
---------------

The long range term is calculated in Fourier space

.. math::
    k^2 \tilde\phi_{\mathcal F}(k) = 4\pi \tilde\rho_{\mathcal F}(k)

where

.. math::
    \tilde\rho_{\mathcal F}(k) = \frac{1}{V} \int d\mathbf re^{- i \mathbf k \cdot \mathbf r} \rho_{\mathcal F}( \mathbf r ) = \sum_{i}^N \frac{q_i\alpha^{3/2}}{\pi V} \int d\mathbf r e^{- i \mathbf k \cdot \mathbf r}  e^{-\alpha^2 \left( \mathbf r - \mathbf r_i \right )^2 } = \sum_{i}^N \frac{q_i}{V} e^{-i \mathbf k \cdot \mathbf r_i} e^{-k^2/(4\alpha^2)}.


The potential is then

.. math::
    \tilde \phi_{\mathcal F}(\mathbf k) = \frac{4\pi}{k^2} \frac{1}{V} \sum_{i}^N q_i e^{-i\mathbf k \cdot \mathbf r_i} e^{-k^2/(4\alpha^2)} = \frac{1}{V} \sum_i^N v(k)e^{-k^2/(4 \alpha^2)} q_i e^{-i \mathbf k \cdot \mathbf r_i}


and in real space

.. math::
    \phi_{\mathcal R}( \mathbf r ) = \sum_{\mathbf k \neq 0} \tilde \phi_{\mathcal F}(\mathbf k)e^{i \mathbf k \cdot \mathbf r} = \frac{1}{V} \sum_{\mathbf k\neq 0} \sum_{i}^N v(k) e^{-k^2/(4\alpha^2)}q_i e^{i \mathbf k \cdot ( \mathbf r- \mathbf r_i) },


where the :math:`\mathbf k = 0` is removed from the sum because of the overall charge neutrality.
The potential energy created by this long range part is

.. math::
    U_{\mathcal F} = \frac {1}{2} \sum_i^N q_i \phi_{\mathcal F}(\mathbf r_i) = \frac{1}{2} \frac{1}{V} \sum_{i,j}^N q_i q_j \sum_{\mathbf k \neq 0 } v(k)  e^{-k^2/(4\alpha^2)}e^{i \mathbf k \cdot ( \mathbf r_i - \mathbf r_j) } = \frac{1}{2} \sum_{\mathbf k \neq 0} |\rho_0(\mathbf k)|^2 v(k) e^{-k^2/(4\alpha^2)},

where I used the definition of the charge density

.. math::
    \rho_0(\mathbf k) = \frac 1V \sum_i^N q_i e^{i \mathbf k \cdot \mathbf r_i}.

However, in the above sum we are including the self-energy term, i.e. :math:`\mathbf r_i = \mathbf r_j`. This term
can be easily calculated and then removed from :math:`U_{\mathcal F}`

.. math::
    \frac{\mathcal Q^2}{2V} \sum_{\mathbf k} \frac{4\pi}{k^2} e^{-k^2/(4\alpha^2)} \rightarrow \frac{\mathcal Q^2}{2V} \left ( \frac{L}{2\pi} \right )^3 \int dk (4\pi)^2 e^{-k^2/(4\alpha^2) }  = \mathcal Q^2 \frac{(4\pi)^2}{2V} \left ( \frac{L}{2\pi} \right )^3 \sqrt{\pi } \alpha = \mathcal Q^2 \frac{\alpha}{\sqrt{\pi} }

where :math:`\mathcal Q^2 = \sum_i^N q_i^2`, note that in the integral we have re-included :math:`\mathbf k = 0`, but
this is not a problem(?). Finally the long-range potential energy is

.. math::

    U_{\mathcal L} = U_{\mathcal F} - \mathcal Q^2 \frac{\alpha}{\sqrt{\pi} }
