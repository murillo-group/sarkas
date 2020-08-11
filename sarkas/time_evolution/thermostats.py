"""
Module containing various thermostat. Berendsen only for now.
"""
import numpy as np
from numba import njit



@njit
def berendsen(vel, T_desired, T, species_np, therm_timestep, tau, it):
    """
    Update particle velocity based on Berendsen thermostat [Berendsen1984]_.

    Parameters
    ----------
    T : array
        Instantaneous temperature of each species.

    vel : array
        Particles' velocities to rescale.

    T_desired : array
        Target temperature of each species.

    tau : float
        Scale factor.

    therm_timestep : int
        Timestep at which to turn on the thermostat.

    species_np : array
        Number of each species.

    it : int
        Current timestep.

    References
    ----------
    .. [Berendsen1984] `H.J.C. Berendsen et al., J Chem Phys 81 3684 (1984) <https://doi.org/10.1063/1.448118>`_

    """
    # Dev Notes: this could be Numba'd
    species_start = 0
    for i in range(len(species_np)):
        species_end = species_start + species_np[i]

        if it <= therm_timestep:
            fact = np.sqrt(T_desired[i] / T[i])
        else:
            fact = np.sqrt(1.0 + (T_desired[i] / T[i] - 1.0) / tau)  # eq.(11)

        vel[species_start:species_end, :] *= fact
        species_start = species_end

    return


