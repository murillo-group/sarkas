"""
Module handling restarting/checkpoint
"""
import os
import csv
from numpy import load, savez, sum
import pickle


class Checkpoint:
    """
    Class to handle restart dumps.

    Parameters
    ----------
    params : class
        Simulation's parameters.

    Attributes
    ----------
    dt : float
        Simulation timestep.

    dump_dir : str
        Path to directory where simulations dump will be stored.

    energy_filename : str
        CSV file for storing energy values.

    ptcls_file_name : str
        Prefix of dumps filenames.

    params_pickle : str
        Pickle file where all simulation parameters will be stored.

    species_names: list
        Names of each particle species.

    checkpoint_dir : str
        Output directory. 
    """

    def __init__(self, params):
        self.dt = params.Control.dt
        self.checkpoint_dir = params.Control.checkpoint_dir
        self.params_pickle = os.path.join(self.checkpoint_dir, "S_parameters.pickle")
        # Production directory and filenames
        self.dump_dir = params.Control.dump_dir
        self.energy_filename = os.path.join(self.checkpoint_dir, "Thermodynamics_" + params.Control.fname_app + '.csv')
        self.ptcls_file_name = os.path.join(self.dump_dir, "S_checkpoint_")
        # Thermalization directory and filenames
        self.therm_dir = params.Control.therm_dir
        self.therm_dump_dir = os.path.join(self.therm_dir, "dumps")
        self.therm_filename = os.path.join(self.therm_dir, "Thermalization_" + params.Control.fname_app + '.csv')
        self.therm_ptcls_file_name = os.path.join(self.therm_dump_dir, "S_checkpoint_")

        self.species_names = []
        self.Gamma_eff = params.Potential.Gamma_eff * params.T_desired

        for i in range(params.num_species):
            self.species_names.append(params.species[i].name)

        # Check the existence of locations
        if not (os.path.exists(self.checkpoint_dir)):
            os.mkdir(self.checkpoint_dir)
        if not (os.path.exists(self.dump_dir)):
            os.mkdir(self.dump_dir)
        if not (os.path.exists(self.therm_dir)):
            os.mkdir(self.therm_dir)

        if not os.path.exists(self.energy_filename):
            # Create the Energy file
            dkeys = ["Time", "Total Energy", "Total Kinetic Energy", "Potential Energy", "Temperature"]
            if len(params.species) > 1:
                for i, sp in enumerate(params.species):
                    dkeys.append("{} Kinetic Energy".format(sp.name))
                    dkeys.append("{} Temperature".format(sp.name))
            dkeys.append("Gamma")
            data = dict.fromkeys(dkeys)

            with open(self.energy_filename, 'w+') as f:
                w = csv.writer(f)
                w.writerow(data.keys())

        if not os.path.exists(self.therm_filename) and not params.load_method == 'restart':
            # Create the Energy file
            dkeys = ["Time", "Total Energy", "Total Kinetic Energy", "Potential Energy", "Temperature"]
            if len(params.species) > 1:
                for i, sp in enumerate(params.species):
                    dkeys.append("{} Kinetic Energy".format(sp.name))
                    dkeys.append("{} Temperature".format(sp.name))
            dkeys.append("Gamma")
            data = dict.fromkeys(dkeys)

            with open(self.therm_filename, 'w+') as f:
                w = csv.writer(f)
                w.writerow(data.keys())

    def save_pickle(self, params):
        """
        Save all simulations parameters in a pickle file.

        Parameters
        ----------
        params : class
            Simulation parameters.

        """
        pickle_file = open(self.params_pickle, "wb")
        pickle.dump(params, pickle_file)
        pickle_file.close()

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
        tme = it * self.dt
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

    def therm_dump(self, ptcls, kinetic_energies, temperatures, potential_energy, it):
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
        fle_name = self.therm_ptcls_file_name + str(it)
        tme = it * self.dt
        savez(fle_name,
              species_id=ptcls.species_id,
              species_name=ptcls.species_name,
              pos=ptcls.pos,
              vel=ptcls.vel,
              acc=ptcls.acc,
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
        with open(self.therm_filename, 'a') as f:
            w = csv.writer(f)
            w.writerow(data.values())