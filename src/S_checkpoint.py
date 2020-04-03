"""
Module handling restarting/checkpoint
"""
import os
import numpy as np
import pickle


class Checkpoint:
    """
    Class to handle restart dumps.

    Attributes
    ----------
    params : class
        Simulation's parameters. see ``S_params.py`` module for more info.
    
    checkpoint_dir : str
        Output directory. 
    """
    def __init__(self, params):
        """
        Save simulation's parameters to a binary file using pickle.

        Parameters
        ----------
        params : class
            Simulation's parameters.
        """
        self.params = params
        # redundancy for the sake of shortness
        self.checkpoint_dir = self.params.Control.checkpoint_dir
        if not (os.path.exists(self.checkpoint_dir)):
            os.mkdir(self.checkpoint_dir)

        filename = self.checkpoint_dir + "/" + "S_parameters.pickle"

        save_file = open(filename, "wb")
        pickle.dump(self.params, save_file)
        save_file.close()

    def dump(self, ptcls, it):
        """
        Save particles' data to binary file for future restart.

        Parameters
        ----------
        ptcls : class
            Particles' data. See ``S_particles.py`` for more info.

        it : int
            Timestep number.
        """
        species_id = ptcls.species_id
        species_name = ptcls.species_name
        pos = ptcls.pos
        vel = ptcls.vel
        acc = ptcls.acc
        cntr = ptcls.pbc_cntr
        file_name = self.checkpoint_dir+"/"+"S_checkpoint_"+str(it)
        np.savez(file_name, species_id=species_id, species_name=species_name, pos=pos, vel=vel, acc=acc, cntr = cntr)
