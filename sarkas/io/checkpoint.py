import os
import csv
import pickle
import numpy as np


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

    def __init__(self, params, species):
        self.dt = params.dt
        self.kB = params.kB
        self.job_dir = params.job_dir
        self.params_pickle = os.path.join(self.job_dir, "simulation_parameters.pickle")
        # Production directory and filenames
        self.production_dir = params.production_dir
        self.prod_dump_dir = params.prod_dump_dir
        self.prod_energy_filename = os.path.join(self.production_dir,
                                                 "ProductionEnergy_" + params.job_id + '.csv')
        self.prod_ptcls_file_name = os.path.join(self.prod_dump_dir, "checkpoint_")
        # Thermalization directory and filenames
        self.equilibration_dir = params.equilibration_dir
        self.eq_dump_dir = params.eq_dump_dir
        self.eq_energy_filename = os.path.join(self.equilibration_dir,
                                               "EquilibrationEnergy_" + params.job_id + '.csv')
        self.eq_ptcls_file_name = os.path.join(self.eq_dump_dir, "checkpoint_")

        self.species_names = params.species_names
        self.coupling = params.coupling_constant * params.T_desired

        if not os.path.exists(self.prod_energy_filename):
            # Create the Energy file
            dkeys = ["Time", "Total Energy", "Total Kinetic Energy", "Potential Energy", "Temperature"]
            if len(species) > 1:
                for i, sp in enumerate(species):
                    dkeys.append("{} Kinetic Energy".format(sp.name))
                    dkeys.append("{} Temperature".format(sp.name))
            dkeys.append("Gamma")
            data = dict.fromkeys(dkeys)

            with open(self.prod_energy_filename, 'w+') as f:
                w = csv.writer(f)
                w.writerow(data.keys())

        if not os.path.exists(self.eq_energy_filename) and not params.load_method[-7:] == 'restart':
            # Create the Energy file
            dkeys = ["Time", "Total Energy", "Total Kinetic Energy", "Potential Energy", "Temperature"]
            if len(species) > 1:
                for i, sp_name in enumerate(params.species_names):
                    dkeys.append("{} Kinetic Energy".format(sp_name))
                    dkeys.append("{} Temperature".format(sp_name))
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

    def dump(self, production, ptcls, it):
        """
        Save particles' data to binary file for future restart.

        Parameters
        ----------
        production: bool
            Flag indicating whether to phase production or equilibration data.

        ptcls: object
            Particles data.

        potential_energy : float
            Potential energy.

        it : int
            Timestep number.
        """
        if production:
            ptcls_file = self.prod_ptcls_file_name + str(it)
            tme = it * self.dt
            np.savez(ptcls_file,
                  id=ptcls.id,
                  names=ptcls.names,
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
            np.savez(ptcls_file,
                  id=ptcls.id,
                  name=ptcls.names,
                  pos=ptcls.pos,
                  vel=ptcls.vel,
                  acc=ptcls.acc,
                  time=tme)

            energy_file = self.prod_energy_filename

        kinetic_energies, temperatures = ptcls.kinetic_temperature(self.kB)
        # Prepare data for saving
        data = {"Time": it * self.dt,
                "Total Energy": np.sum(kinetic_energies) + ptcls.potential_energy,
                "Total Kinetic Energy": np.sum(kinetic_energies),
                "Potential Energy": ptcls.potential_energy,
                "Total Temperature": np.sum(temperatures)
                }
        for sp, kin in enumerate(kinetic_energies):
            data["{} Kinetic Energy".format(self.species_names[sp])] = kin
            data["{} Temperature".format(self.species_names[sp])] = temperatures[sp]
        with open(energy_file, 'a') as f:
            w = csv.writer(f)
            w.writerow(data.values())
