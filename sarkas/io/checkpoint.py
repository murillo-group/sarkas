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

    Parameters
    ----------
    params: object
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

    job_dir : str
        Output directory. 
    """

    def __init__(self, params):
        self.dt = params.integrator.dt
        self.job_dir = params.control.job_dir
        self.params_pickle = os.path.join(self.job_dir, "S_parameters.pickle")
        # Production directory and filenames
        self.production_dir = params.control.production_dir
        self.prod_dump_dir = params.control.prod_dump_dir
        self.prod_energy_filename = os.path.join(self.production_dir,
                                                              "ProductionEnergies_" + params.control.job_id + '.csv')
        self.prod_ptcls_file_name = os.path.join(self.prod_dump_dir, "S_checkpoint_")
        # Thermalization directory and filenames
        self.equilibration_dir = params.control.equilibration_dir
        self.eq_dump_dir = params.control.eq_dump_dir
        self.eq_energy_filename = os.path.join(self.equilibration_dir,
                                                              "EquilibrationEnergies_" + params.control.job_id + '.csv')
        self.eq_ptcls_file_name = os.path.join(self.eq_dump_dir, "S_checkpoint_")

        self.species_names = []
        self.Gamma_eff = params.potential.Gamma_eff * params.T_desired

        for sp in params.species:
            self.species_names.append(sp.name)

        if not os.path.exists(self.prod_energy_filename):
            # Create the Energy file
            dkeys = ["Time", "Total Energy", "Total Kinetic Energy", "Potential Energy", "Temperature"]
            if len(params.species) > 1:
                for i, sp in enumerate(params.species):
                    dkeys.append("{} Kinetic Energy".format(sp.name))
                    dkeys.append("{} Temperature".format(sp.name))
            dkeys.append("Gamma")
            data = dict.fromkeys(dkeys)

            with open(self.prod_energy_filename, 'w+') as f:
                w = csv.writer(f)
                w.writerow(data.keys())

        if not os.path.exists(self.eq_energy_filename) and not params.control.restart == 'restart':
            # Create the Energy file
            dkeys = ["Time", "Total Energy", "Total Kinetic Energy", "Potential Energy", "Temperature"]
            if len(params.species) > 1:
                for i, sp in enumerate(params.species):
                    dkeys.append("{} Kinetic Energy".format(sp.name))
                    dkeys.append("{} Temperature".format(sp.name))
            dkeys.append("Gamma")
            data = dict.fromkeys(dkeys)

            with open(self.eq_energy_filename, 'w+') as f:
                w = csv.writer(f)
                w.writerow(data.keys())

    def save_pickle(self, params):
        """
        Save all simulations parameters in a pickle file.

        Parameters
        ----------
        params: object
            Simulation's parameters.

        """
        pickle_file = open(self.params_pickle, "wb")
        pickle.dump(params, pickle_file)
        pickle_file.close()

    def dump(self, production, ptcls, kinetic_energies, temperatures, potential_energy, it):
        """
        Save particles' data to binary file for future restart.

        Parameters
        ----------
        production: bool
            Flag indicating whether to phase production or equilibration data.

        ptcls: object
            Particles data.

        kinetic_energies : array 
            Kinetic energy of each species.
        
        temperatures : array 
            Temperature of each species.
        
        potential_energy : float
            Potential energy.
            
        it : int
            Timestep number.
        """
        if production:
            ptcls_file = self.prod_ptcls_file_name + str(it)
            tme = it * self.dt
            savez(ptcls_file,
                  species_id=ptcls.species_id,
                  species_name=ptcls.species_name,
                  pos=ptcls.pos,
                  vel=ptcls.vel,
                  acc=ptcls.acc,
                  cntr=ptcls.pbc_cntr,
                  rdf_hist=ptcls.rdf_hist,
                  time=tme)

            energy_file = self.prod_energy_filename

        else:
            ptcls_file = self.prod_ptcls_file_name + str(it)
            tme = it * self.dt
            savez(ptcls_file,
                  species_id=ptcls.species_id,
                  species_name=ptcls.species_name,
                  pos=ptcls.pos,
                  vel=ptcls.vel,
                  acc=ptcls.acc,
                  time=tme)

            energy_file = self.prod_energy_filename

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
        with open(energy_file, 'a') as f:
            w = csv.writer(f)
            w.writerow(data.values())
