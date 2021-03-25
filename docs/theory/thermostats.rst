===========
Thermostats
===========

Berendsen Thermostat
====================

The Berendsen Thermostat (BT) uses a tapered velocity scaling approach. In a strict velocity scaling approach
the temperature :math:`T_e` is estimated, through a quantity proportional to :math:`\langle v^2 \rangle`,
and the velocities are scaled to values consistent with the desired temperature :math:`T_d`,
as in :math:`v_i \to \alpha v_i`. Being completely inconsistent with physical laws,
it is preferable to use the same simple algorithm but more gently so that the dynamics during the thermostat period is
more consistent with the underlying equations of motion.
In the BT we begin with an model for the temperature as we would like to see it evolve over a slower
timescale :math:`\tau_{B}`. One model is

.. math::
   \frac{dT}{dt} = \frac{T_d - T}{\tau_{B}},

This equation can be solved analytically to yield

.. math::
   T(t) = T(0)e^{-t/\tau_B} + \left(1 - e^{-t/\tau_B}  \right)T_d ,

which can be seen to transition from the initial temperature :math:`T(0)` to the desired temperature :math:`T_d`
on a time scale of :math:`\tau_{B}`. By choosing :math:`\tau_{B}` to be many time steps we can eventually equilibrate
the system while allowing it to explore configurations closer to the real (not velocity scaled) dynamics.

To implement BT we discretize the BT model across one time step to obtain

.. math::
   T(t + \Delta t) = T(t) + \frac{\Delta t}{\tau_B}\left(T_d - T(t) \right) .

We want to scale the current velocities such that this new temperature :math:`T(t+\Delta t)` is achieved,
because that the temperature prescribed by the BT.
Finding the ratio then of the target temperature and the current temperature, we get

.. math::
   \frac{T(t + \Delta t)}{T(t) } = 1+ \frac{\Delta t}{\tau_{B}}\left(\frac{T_d}{T(t) } - 1 \right)

Taking the square root this yields the scaling factor for the velocities.

The BT is implemented in Sarkas with several epochs. You can specify the number of epochs and the individual
strengths :math:`\tau_{B,j}` (What is this ?). This allows for a stronger thermostat initially,
but tapering to a more gentle thermostat toward the end of the equilibration phase.

Below ww show a plot of the temperature difference

.. math::

    \Delta T(t) = T(t) - T_d

for several values of :math:`\tau_B`. As you can see the smaller :math:`\tau_B` the faster :math:`T(t)` reaches the
desired temperature.