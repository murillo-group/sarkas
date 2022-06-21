.. _integrators:

===========
Integrators
===========
Sarkas aims to support a variety of time integrators both built-in and user defined.
Currently the available ones are:

- :ref:`Velocity Verlet <vel_verlet>`
- Langevin dynamics
- Magnetic Position Verlet
- :ref:`Magnetic Velocity Verlet <mag_vel_verlet>`
- Magnetic Boris
- Cyclotronic

The choice of integrator is provided in the input file and the method
:doc:`type_setup() <../api/time_evolution_subpckg/Integrator_mthds/sarkas.time_evolution.integrators.Integrator.type_setup>`
links the chosen integrator to the :doc:`update() <../api/time_evolution_subpckg/Integrator_mthds/sarkas.time_evolution.integrators.Integrator.update>` method which evolves
particles' positions, velocities, and accelerations in time.

The Velocity Verlet algorithm is the most common integrator used in MD plasma codes.
It is preferred to other more accurate integrator, such as RK45, inasmuch as it conserves the symmetries of the
Hamiltonian, it is fast and easy to implement.

Phase Space Distribution
------------------------
The state of the system is defined by the set of phase space coordinates
:math:`\{ \mathbf r, \mathbf p \} = \{ \mathbf r_1, \mathbf r_2, \dots, \mathbf r_N , \mathbf p_1, \mathbf p_2, \dots, \mathbf p_N \}`
where :math:`N` represents the number of particles. The system evolves in time according to the Hamiltonian

.. math::
    \mathcal H = \mathcal T + \mathcal U,

where :math:`\mathcal T` is the kinetic energy and :math:`\mathcal U` the interaction potential. The :math:`N`-particle
probability distribution :math:`f_N(\mathbf r, \mathbf p; t)` evolves in time according to the Liouville equation

.. math::
    i\mathcal L f_N(\mathbf r, \mathbf p;0) = 0,

with

.. math::
    \mathcal L = \frac{\partial}{\partial t} + \dot{\mathbf r} \cdot \frac{\partial}{\partial \mathbf r} + \dot{\mathbf p}\cdot \frac{\partial}{\partial \mathbf p},

.. math::
    \dot{\mathbf r} = \frac{\partial \mathcal H}{\partial \mathbf p}, \quad \dot{\mathbf p} = - \frac{\partial \mathcal H}{\partial \mathbf r}.

The solution of the Liouville equation is

.. math::
    \mathcal f_N(\mathbf r, \mathbf p;t) =  e^{- i \mathcal L t } f_N(\mathbf r, \mathbf p;0)

.. _vel_verlet:
Velocity Verlet
---------------
It can be shown that the Velocity Verlet corresponds to a second order splitting of the Liouville operator :math:`\mathcal L =  K +  V`

.. math::
    e^{i \epsilon \mathcal L} \approx e^{\frac{\Delta t}{2} K}e^{\Delta t V}e^{\frac{\Delta t}{2} K}

where :math:`\epsilon = -i \Delta t` and the operators

.. math::
    K = \mathbf v \cdot \frac{\partial}{\partial \mathbf r}, \quad
    V = \mathbf a \cdot \frac{\partial}{\partial \mathbf v}.

Any dynamical quantities :math:`W` evolves in time according to the Liouville operator :math:`\mathcal L =  K +  V`

.. math::
    W(t) = e^{i\epsilon (K +  V)} W(0).


Applying each one of these to the initial set :math:`\mathbf W = ( \mathbf r_0, \mathbf v_0)` we find
:math:`e^{i\epsilon V} \mathbf r_0 = e^{i\epsilon K} \mathbf v_0 = 0` and

.. math::
    e^{i \epsilon K} \mathbf r_0 & = &  \left ( 1 + \Delta t K - \frac{\Delta t^2 K^2}{2!} + \frac{\Delta t^3 K^3}{3!} + ... \right ) \mathbf r_0 \nonumber \\  & = & \left [ 1 + \Delta t \mathbf v \cdot \frac{\partial}{\partial \mathbf r} - \frac{\Delta t^2}{2} \left ( \mathbf v \cdot \frac{\partial}{\partial \mathbf r} \right )^2 + ... \right ] \mathbf r_0 ,

.. math::
    e^{i \epsilon V} \mathbf v_0 & = &  \left ( 1 + \Delta t V - \frac{\Delta t^2 V^2}{2!} + \frac{\Delta t^3 V^3}{3!} + ... \right ) \mathbf r_0 \nonumber \\  & = & \left [ 1 + \Delta t \mathbf a \cdot \frac{\partial}{\partial \mathbf v} - \frac{\Delta t^2}{2} \left ( \mathbf a \cdot \frac{\partial}{\partial \mathbf v} \right )^2 + ... \right ] \mathbf v_0.

Only the first two terms in the square brackets survive as higher order derivative vanish, thus leading to the update
equations

.. math::
    \mathbf r(t + \Delta t) = \mathbf r_0 + \Delta t \mathbf v(t), \quad \mathbf v(t + \Delta t) = \mathbf v_0 + \Delta t \mathbf a(t).

.. _mag_vel_verlet:
Magnetic Velocity Verlet
------------------------
A generalization to include constant external magnetic fields leads to the Liouville operator
:math:`e^{i \epsilon( K + V + L_B)}` where :cite:`Chin2008`

.. math::
    L_B = \omega_c \left ( \hat{\mathbf B} \times \mathbf v \right ) \cdot \frac{\partial}{\partial \mathbf v}  = \omega_c \hat{\mathbf B} \cdot \left( \mathbf v \times \frac{\partial}{\partial \mathbf v} \right ) = \omega_c \hat{\mathbf B} \cdot \mathbf J_{\mathbf v}.

Application of this operator leads to :math:`e^{i \epsilon L_B}\mathbf{r}_0 = 0` and

.. math::
    e^{ i \epsilon L_B } \mathbf v_0 & = &  \left ( 1 + \Delta t V - \frac{\Delta t^2 V^2}{2!} + \frac{\Delta t^3 V^3}{3!} + ... \right ) \mathbf v_0 \nonumber \\  & = & \left [ 1 + \omega_c \Delta t  \hat{\mathbf B} \cdot \mathbf J_{\mathbf v} - \frac{\omega_c^2 \Delta t^2}{2}  \left ( \hat{\mathbf B} \cdot \mathbf J_{\mathbf v} \right )^2 + ... \right ] \mathbf v_0 \nonumber \\
    & = & \begin{pmatrix}
    \cos(\omega_c\Delta t) & - \sin(\omega_c\Delta t) & 0 \\
    \sin(\omega_c\Delta t) & \cos(\omega_c\Delta t) & 0 \\
    0 & 0 & 1 \\
    \end{pmatrix} \mathbf v_0 \\
    & = &\mathbf v_{0,\parallel} + \cos(\omega_c \Delta t) \mathbf v_{0,\perp} + \sin(\omega_c \Delta t) \hat{\mathbf B} \times \mathbf v_{0, \perp},

where in the last passage we have divided the velocity in its parallel and perpendicular component to the
:math:`\\mathbf B` field. In addition, we have

.. math::
    e^{i \epsilon (L_B + V) } \mathbf v_0 & = & e^{i \epsilon L_B} \mathbf v_0 + \Delta t \mathbf a + \frac{1 - \cos(\omega_c \Delta t)}{\omega_c} \left ( \hat{\mathbf B} \times \mathbf a \right ) \nonumber \\
    && + \Delta t \left ( 1 - \frac{\sin(\omega_c \Delta t)}{\omega_c \Delta t} \right ) \left [ \hat {\mathbf B} \times \left ( \hat{\mathbf B} \times \mathbf a \right ) \right ].

Time integrators of various order can be found by exponential splitting, that is

.. math::
    e^{i \epsilon \mathcal L} \approx \prod_{ j = 1}^{N} e^{i a_j \epsilon K} e^{i b_j \epsilon \left ( L_B + V \right ) }.

The Boris algorithm, widely used in Particle in Cell simulations, corresponds to :cite:`Chin2008`

.. math::
   e^{i \epsilon \mathcal L} \approx e^{i \epsilon K} e^{i \epsilon V/2}  e^{i \epsilon L_B} e^{i \epsilon V/2}

while a generalization of the Velocity-Verlet :cite:`Chin2008,Spreiter1999`

.. math::
   e^{i \epsilon \mathcal L} \approx  e^{i \epsilon (L_B + V) /2} e^{i \epsilon K} e^{i \epsilon ( L_B + V)/2}.

Notice that all the above algorithm require one force calculation per time step.
