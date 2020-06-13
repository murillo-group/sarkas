"""
Module handling restarting/checkpoint
"""
import os
import csv
from numpy import savez, sum
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
        self.dt = params.Control.dt
        self.checkpoint_dir = params.Control.checkpoint_dir
        self.ptcls_dir = os.path.join(self.checkpoint_dir, "Particles_Data")
        self.energy_filename = os.path.join( self.checkpoint_dir, "Thermodynamics_" + params.Control.fname_app + '.csv')
        self.ptcls_file_name = os.path.join( self.ptcls_dir,  "S_checkpoint_")
        self.species_names = []
        self.Gamma_eff = params.Potential.Gamma_eff * params.T_desired

        for i in range(params.num_species):
            self.species_names.append(params.species[i].name)

        if not (os.path.exists(self.checkpoint_dir)):
            os.mkdir(self.checkpoint_dir)
        if not (os.path.exists(self.ptcls_dir)):
            os.mkdir(self.ptcls_dir)

        save_file = open(os.path.join(self.checkpoint_dir, "S_parameters.pickle"), "wb")
        pickle.dump(params, save_file)
        save_file.close()

        if not params.load_method == "restart":
            data = {"Time": [],
                    "Total Energy": [],
                    "Total Kinetic Energy": [],
                    "Potential Energy": [],
                    "Temperature": []}
            if params.num_species > 1:
                for sp in range(params.num_species):
                    data["{} Kinetic Energy".format(self.species_names[sp])] = []
                    data["{} Temperature".format(self.species_names[sp])] = []
            data["Gamma"] = []
            with open(self.energy_filename, 'w') as f:
                w = csv.writer(f)
                w.writerow(data.keys())

    def dump(self, ptcls, kinetic_energies, temperatures, potential_energy, it):
        """
        Save particles' data to binary file for future restart.

        Parameters
        ----------
        ptcls : class
            Particles' data. See ``S_particles.py`` for more info.
        
        kinetic_energies : array 
            Kinetic energy of each species.
        
        temperatures : array 
            Temperature of each species.
        
        potential_energy : float
            Potential energy.
            
        it : int
            Timestep number.
        """
        fle_name = self.ptcls_file_name + str(it)
        tme = it*self.dt
        savez(fle_name,
              species_id=ptcls.species_id,
              species_name=ptcls.species_name,
              pos=ptcls.pos,
              vel=ptcls.vel,
              acc=ptcls.acc,
              cntr=ptcls.pbc_cntr,
              rdf_hist=ptcls.rdf_hist,
              time=tme)

        # Prepare data for saving
        data = {"Time": it * self.dt,
                "Total Energy": sum(kinetic_energies) + potential_energy,
                "Total Kinetic Energy": sum(kinetic_energies),
                "Potential Energy": potential_energy}
        temperature = ptcls.species_conc.transpose() @ temperatures
        data["Temperature"] = temperature
        if len(temperatures) > 1:
            for sp in range(len(temperatures)):
                data["{} Kinetic Energy".format(self.species_names[sp])] = kinetic_energies[sp]
                data["{} Temperature".format(self.species_names[sp])] = temperatures[sp]
        data["Gamma"] = self.Gamma_eff / temperature
        with open(self.energy_filename, 'a') as f:
            w = csv.writer(f)
            w.writerow(data.values())
