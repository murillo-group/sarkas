"""
Module handling stages of an MD run: PreProcessing, Simulation, PostProcessing.
"""

from importlib import import_module
from IPython import get_ipython
from threading import Thread

if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm import tqdm_notebook as tqdm
    from tqdm.notebook import trange
else:
    from tqdm import tqdm, trange

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import LogNorm
from numpy import (
    arange,
    array,
    int64,
    linspace,
    log2,
    log10,
    logspace,
    meshgrid,
    sqrt,
    zeros,
)
from os import listdir, mkdir
from os import remove as os_remove
from os import stat as os_stat
from os.path import exists, join
from pandas import DataFrame, read_csv
from seaborn import scatterplot
from warnings import warn

from .core import Parameters
from .particles import Particles
from .plasma import Species
from .potentials.core import Potential
from .time_evolution.integrators import Integrator

# Sarkas modules
from .utilities.io import InputOutput, print_to_logger
from .utilities.maths import force_error_analytic_pp, force_error_approx_pppm
from .utilities.timing import SarkasTimer


class Process:
    """Parent class for :class:`sarkas.process.PreProcess`, :class:`sarkas.process.Simulation`, and
    :class:`sarkas.process.PostProcess`.

    Parameters
    ----------
    input_file : str
        Path to the YAML input file. Default = `None`

    Attributes
    ----------
    potential : :class:`sarkas.potential.base.Potential`
        Class handling the interaction between particles.

    integrator : :class:`sarkas.time_evolution.integrators.Integrator`
        Class handling the integrator.

    particles: :class:`sarkas.particles.Particles`
        Class handling particles properties.

    parameters : :class:`sarkas.core.Parameters`
        Class handling simulation's parameters.

    species : list
        List of :class:`sarkas.plasma.Species` classes.

    input_file : str
        Path to YAML input file.

    timer : :class:`sarkas.utilities.timing.SarkasTimer`
        Class handling the timing of processes.

    io : :class:`sarkas.utilities.io.InputOutput`
        Class handling the IO in Sarkas.

    """

    from .potentials.core import Potential

    def __init__(self, 
                 input_file: str = None, 
                 potential_class: Potential = None,
                 integrator_class: Integrator = None,
                 particles_class: Particles = None,
                 parameters_class: Parameters = None,
                 io_class: InputOutput = None,
                 species: list = None,
                 
                 ):
        
        if potential_class is not None:
            self.potential = potential_class
        else:
            self.potential = Potential()

        if integrator_class is not None:
            self.integrator = integrator_class
        else:
            self.integrator = Integrator()

        if particles_class is not None:
            self.particles = particles_class
        else:   
            self.particles = Particles()
        
        if parameters_class is not None:
            self.parameters = parameters_class
        else:
            self.parameters = Parameters()
        
        if io_class is not None:
            self.io = io_class
        else:
            self.io = InputOutput(process=self.__name__)

        if species is not None:
            self.species = species
        else:
            self.species = []
    
        self.threads_ls = []
        self.observables_dict = {}
        self.transport_dict = {}

        self.input_file = input_file
        self.timer = SarkasTimer()

    def common_parser(self, filename: str = None):
        """
        Parse simulation parameters from a YAML file.

        Parameters
        ----------
        filename : str, optional
            Path to the YAML input file. If not provided, the input file path specified during object initialization will be used.

        Returns
        -------
        dict
            A nested dictionary containing the parsed simulation parameters.

        Notes
        -----
        This method reads the simulation parameters from a YAML file and returns them as a nested dictionary. 
        It uses the :meth:`sarkas.utilities.io.InputOutput.from_yaml` to read the YAML file.

        If the `filename` parameter is provided, it will override the input file path specified during object initialization.

        Examples
        --------
        >>> process = Process(input_file='/path/to/input.yaml')
        >>> params_dict = process.common_parser()
        
        """
        if filename:
            self.input_file = filename

        params_dict = self.io.from_yaml(self.input_file)

        return params_dict

    def update_subclasses_from_dict(self, nested_dict: dict):
        """Update the subclasses parameters using a dictionary.

        Parameters
        ----------
        nested_dict : dict
            Nested dictionary. See example for format.

        """
        for lkey, vals in nested_dict.items():
            if lkey not in ["Particles", "Observables", "TransportCoefficients"]:
                self.__dict__[lkey.lower()].__dict__.update(vals)
            elif lkey in ["Particles", "Plasma"]:
                # Remember Particles should be a list of dict
                # example:
                # args = {"Particles" : [ { "Species" : { "name": "O" } } ] }

                # Check if you already have a non-empty dict of species
                if len(self.species) > 0:
                    # If so do you want to replace or update?
                    # Update species attributes

                    for sp, species in enumerate(vals):
                        spec = Species(species["Species"])
                        if hasattr(spec, "replace"):
                            self.species[sp].__dict__.update(spec.__dict__)
                        else:
                            self.species.append(spec)
                else:
                    # Append new species
                    for sp, species in enumerate(vals):
                        spec = Species(species["Species"])
                        self.species.append(spec)

            elif lkey in ["Observables"]:
                for obs_dict in vals:
                    for obs, params in obs_dict.items():
                        module = import_module(".observables", "sarkas.tools")
                        class_ = getattr(module, obs)
                        inst = class_()
                        if inst.__long_name__ in self.observables_dict.keys():
                            self.observables_dict[inst.__long_name__].__dict__.update(params)
                        else:
                            self.observables_dict[inst.__long_name__] = inst
                            self.observables_dict[inst.__long_name__].from_dict(params)

            elif lkey in ["TransportCoefficients"]:
                for obs_dict in vals:
                    for obs, params in obs_dict.items():
                        module = import_module(".transport", "sarkas.tools")
                        class_ = getattr(module, obs)
                        inst = class_()
                        if inst.__long_name__ in self.transport_dict.keys():
                            self.transport_dict[inst.__long_name__].__dict__.update(params)
                        else:
                            self.transport_dict[inst.__long_name__] = inst
                            self.transport_dict[inst.__long_name__].from_dict(params)

        # electron properties has been moved to the Parameters class. Therefore I need to put this here.
        if hasattr(self.potential, "electron_temperature"):
            self.parameters.electron_temperature = self.potential.electron_temperature
        elif hasattr(self.potential, "electron_temperature_eV"):
            self.parameters.electron_temperature_eV = self.potential.electron_temperature_eV

    def instantiate_subclasses_from_dict(self, nested_dict: dict):
        """Instantiate the process subclasses from a dictionary."""

        for lkey, vals in nested_dict.items():
            if lkey not in ["Particles", "Observables", "TransportCoefficients"]:
                # self.__dict__[lkey.lower()].__dict__.update(vals)

                if lkey == "Potential":
                    self.potential.from_dict(nested_dict[lkey])

                elif lkey == "Integrator":
                    self.integrator.from_dict(nested_dict[lkey])

                elif lkey == "Parameters":
                    self.parameters.from_dict(nested_dict[lkey])

            elif lkey in ["Particles", "Plasma"]:
                for sp, species in enumerate(vals):
                    spec = Species(species["Species"])
                    self.species.append(spec)

            elif lkey in ["Observables"]:
                for obs_dict in vals:
                    for obs, params in obs_dict.items():
                        module = import_module(".observables", "sarkas.tools")
                        class_ = getattr(module, obs)
                        inst = class_()
                        self.observables_dict[inst.__long_name__] = inst
                        self.observables_dict[inst.__long_name__].from_dict(params)

            elif lkey in ["TransportCoefficients"]:
                for obs_dict in vals:
                    for obs, params in obs_dict.items():
                        module = import_module(".transport", "sarkas.tools")
                        class_ = getattr(module, obs)
                        inst = class_()
                        self.transport_dict[inst.__long_name__] = inst
                        self.transport_dict[inst.__long_name__].from_dict(params)

        # electron properties has been moved to the Parameters class. Therefore I need to put this here.
        if hasattr(self.potential, "electron_temperature"):
            self.parameters.electron_temperature = self.potential.electron_temperature
        elif hasattr(self.potential, "electron_temperature_eV"):
            self.parameters.electron_temperature_eV = self.potential.electron_temperature_eV

    def directory_sizes(self):
        """Calculate the size of the dumps directories and print them to logger."""
        # Estimate size of dump folder
        # Grab one file from the dump directory and get the size of it.
        if self.parameters.equilibration_phase:
            if not listdir(self.io.eq_dump_dir):
                raise FileNotFoundError(
                    "Could not estimate the size of the equilibration phase dumps"
                    " because there are no dumps in the equilibration directory."
                    "Re-run .time_n_space_estimate(loops) with loops > eq_dump_step"
                )
            else:
                eq_dump_size = os_stat(join(self.io.eq_dump_dir, listdir(self.io.eq_dump_dir)[0])).st_size
                eq_dump_fldr_size = eq_dump_size * (self.parameters.equilibration_steps / self.parameters.eq_dump_step)
        else:
            eq_dump_size = 0
            eq_dump_fldr_size = 0

        if not listdir(self.io.prod_dump_dir):
            raise FileNotFoundError(
                "Could not estimate the size of the production phase dumps because"
                " there are no dumps in the production directory."
                "Re-run .time_n_space_estimate(loops) with loops > prod_dump_step"
            )

        # Grab one file from the dump directory and get the size of it.
        prod_dump_size = os_stat(join(self.io.prod_dump_dir, listdir(self.io.prod_dump_dir)[0])).st_size
        prod_dump_fldr_size = prod_dump_size * (self.parameters.production_steps / self.parameters.prod_dump_step)
        # Prepare arguments to pass for print out
        sizes = array([[eq_dump_size, eq_dump_fldr_size], [prod_dump_size, prod_dump_fldr_size]])
        # Check for electrostatic equilibration
        if self.parameters.magnetized and self.parameters.electrostatic_equilibration:
            if not listdir(self.io.mag_dump_dir):
                raise FileNotFoundError(
                    "Could not estimate the size of the magnetization phase dumps because"
                    " there are no dumps in the production directory."
                    "Re-run .time_n_space_estimate(loops) with loops > mag_dump_step"
                )
            dump = self.parameters.mag_dump_step
            mag_dump_size = os_stat(join(self.io.mag_dump_dir, "checkpoint_" + str(dump) + ".npz")).st_size
            mag_dump_fldr_size = mag_dump_size * (self.parameters.magnetization_steps / self.parameters.mag_dump_step)
            sizes = array(
                [
                    [eq_dump_size, eq_dump_fldr_size],
                    [prod_dump_size, prod_dump_fldr_size],
                    [mag_dump_size, mag_dump_fldr_size],
                ]
            )

        self.io.directory_size_report(sizes, process=self.__name__)

    def evolve(self, phase, thermalization, it_start, it_end, dump_step):
        """
        Evolve the system forward in time.

        Parameters
        ----------
        phase: str
            Indicates the stage of the simulation used for saving dumps in the right directory. \n
            Choices = ("equilibration", "production", "magnetization")

        thermalization : bool
            Indicates whether to apply the thermostat or not.

        it_start: int
            Initial timestep of the loop.

        it_end: int
            Final timestep of the loop.

        dump_step: int
            Interval for dumping data.

        """

        for it in trange(it_start, it_end, disable=not self.parameters.verbose):
            # Calculate the Potential energy and update particles' data

            self.integrator.update(self.particles)

            if (it + 1) % dump_step == 0:
                self.particles.calculate_observables()
                self.io.dump(phase, self.particles, it + 1)

            if thermalization and (it + 1 >= self.integrator.thermalization_timestep):
                self.particles.calculate_species_kinetic_temperature()
                self.integrator.thermostate(self.particles)

    def evolve_loop_threading(self, phase, thermalization, it_start, it_end, dump_step):
        """
        Evolve the system forward in time. This method is similar to :meth:`sarkas.processes.Process.evolve_loop` with
        the only difference that it uses `threading` for saving data, it starts a new thread to save the data.
        In the case of small number of particles this can slow down the simulation, therefore it must be chosen by setting
        the parameters `threading = True` in the input file or in the :class:`sarkas.core.Parameters` class.

        Parameters
        ----------
        phase: str
            Indicates the stage of the simulation used for saving dumps in the right directory. \n
            Choices = ("equilibration", "production", "magnetization")

        thermalization : bool
            Indicates whether to apply the thermostat or not.

        it_start: int
            Initial timestep of the loop.

        it_end: int
            Final timestep of the loop.

        dump_step: int
            Interval for dumping data.

        """
        for it in trange(it_start, it_end, disable=not self.parameters.verbose):
            # Calculate the Potential energy and update particles' data

            self.integrator.update(self.particles)

            if (it + 1) % dump_step == 0:
                th = Thread(
                    target=self.io.dump,
                    name=f"Sarkas_{phase.capitalize()}_Thread - {it + 1}",
                    args=(
                        phase,
                        self.particles.__deepcopy__(),
                        it + 1,
                    ),
                )

                self.threads_ls.append(th)

                th.start()

            if thermalization and (it + 1 >= self.integrator.thermalization_timestep):
                self.integrator.thermostate(self.particles)

        # Wait for all the threads to finish
        for x in self.threads_ls:
            x.join()

        self.threads_ls.clear()

    def initialization(self):
        """Initialize all classes."""

        # initialize the directories and filenames
        self.io.setup()

        # Copy relevant subsclasses attributes into parameters class. This is needed for post-processing.
        # it updates parameters' dictionary with filenames and directories
        self.parameters.copy_io_attrs(self.io)

        self.parameters.potential_type = self.potential.type.lower()
        self.parameters.setup(self.species)
        self.parameters.dt = self.integrator.dt
        # Initialize particles
        t0 = self.timer.current()
        self.particles.setup(self.parameters, self.species)
        time_ptcls = self.timer.current()

        # Initialize potential and calculate initial potential
        self.potential.setup(self.parameters, self.species)
        self.potential.calc_acc_pot(self.particles)
        time_pot = self.timer.current()
        self.parameters.cutoff_radius = self.potential.rc

        # Initialize Integrator
        self.integrator.setup(self.parameters, self.potential)

        # Copy needed parameters for pretty print
        self.parameters.dt = self.integrator.dt
        self.parameters.equilibration_integrator = self.integrator.equilibration_type
        self.parameters.production_integrator = self.integrator.production_type
        if self.parameters.magnetized:
            self.parameters.magnetization_integrator = self.integrator.magnetization_type

        # Copy some parameters needed for saving data
        self.io.copy_params(self.parameters)
        # For restart and backups.
        self.io.setup_checkpoint(self.parameters)
        self.io.save_pickle(self)

        # Print Process summary to file and screen
        self.io.simulation_summary(self)
        time_end = self.timer.current()

        # self.evolve = self.evolve_loop_threading if self.parameters.threading else self.evolve_loop

        # Print timing
        self.io.time_stamp("Particles Initialization", self.timer.time_division(time_ptcls - t0))
        self.io.time_stamp("Potential Initialization", self.timer.time_division(time_pot - time_ptcls))
        self.io.time_stamp("Total Simulation Initialization", self.timer.time_division(time_end - t0))

        self.print_initial_state()

    def print_initial_state(self):
        """Print the initial energies of the system."""

        init_eng = " Initial Energies "
        msg = f"\n\n{init_eng:-^70}\n" f"Initial temperature and kinetic energy of each species\n"

        self.particles.calculate_species_kinetic_temperature()
        self.particles.calculate_species_potential_energy()

        factor = self.parameters.J2erg if self.parameters.units == "mks" else 1.0 / self.parameters.J2erg

        for sp, kp, tp, pot_sp in zip(
            self.species,
            self.particles.species_kinetic_energy,
            self.particles.species_temperatures,
            self.particles.species_potential_energy,
        ):
            sp_msg = (
                f"Species {sp.name} :\n"
                f"\tTemperature = {tp:.6e} {self.parameters.units_dict['temperature']} = {tp * self.parameters.eV2K:.6e} {self.parameters.units_dict['electron volt']}\n"
                f"\tKinetic Energy = {kp:.6e} {self.parameters.units_dict['energy']} = {kp * factor / self.parameters.eV2J:.6e} {self.parameters.units_dict['electron volt']}\n"
                f"\tPotential Energy = {pot_sp:.6e} {self.parameters.units_dict['energy']} = {pot_sp * factor / self.parameters.eV2J:.6e} {self.parameters.units_dict['electron volt']}\n"
            )
            msg += sp_msg

        tot_kin_e = self.particles.species_kinetic_energy.sum()
        tot_pot_e = self.particles.species_potential_energy.sum()
        tot_e = tot_kin_e + tot_pot_e

        msg += (
            f"Initial total kinetic energy = {tot_kin_e:.6e} {self.parameters.units_dict['energy']} = {tot_kin_e * factor / self.parameters.eV2J:.6e} {self.parameters.units_dict['electron volt']}\n"
            f"Initial total potential energy = {tot_pot_e:.6e} {self.parameters.units_dict['energy']} = {tot_pot_e * factor / self.parameters.eV2J:.6e} {self.parameters.units_dict['electron volt']}\n"
            f"Initial total energy = {tot_e:.6e} {self.parameters.units_dict['energy']} = {tot_e * factor / self.parameters.eV2J:.6e} {self.parameters.units_dict['electron volt']}\n"
        )
        self.io.write_to_logger(msg)

    def setup(self, read_yaml: bool = True, input_file: str = None, other_inputs: dict = None):
        """
        Setup simulations' subclasses by reading the YAML input file and update them with `other_inputs`.

        Parameters
        ----------
        read_yaml: bool
            Flag for reading YAML input file. Default = True.

        input_file: str (optional)
            Path to YAML file with inputs.

        other_inputs: dict (optional)
            Nested dictionary with additional simulations options. This is called after reading the YAML file.

        """
        if input_file:
            self.input_file = input_file

        if read_yaml:
            yaml_dict = self.common_parser()
            self.instantiate_subclasses_from_dict(yaml_dict)

        if other_inputs:
            if not isinstance(other_inputs, dict):
                raise TypeError("Wrong input type. other_inputs should be a nested dictionary")
            self.update_subclasses_from_dict(other_inputs)

        if self.__name__ == "postprocessing":
            # Create the file paths without creating directories and redefining io attributes
            self.io.make_directory_tree()
            self.io.make_directories()
            self.io.make_files_tree()

            # Read previously stored files
            self.io.read_pickle(self, self.io.directory_tree["simulation"]["path"])
            self.io.copy_params(self.parameters)
            # Print parameters to log file
            if not exists(self.io.log_file):
                # if the file exists do not print the file header
                self.io.file_header()
                self.io.simulation_summary(self)

            self.io.datetime_stamp()

            if self.grab_last_step:
                # Initialize the Particles class attributes by reading the last step
                old_method = self.parameters.load_method
                self.parameters.load_method = "production_restart"
                no_dumps = len(listdir(self.io.prod_dump_dir))
                last_step = self.parameters.prod_dump_step * (no_dumps - 1)
                if no_dumps == 0:
                    self.parameters.load_method = "equilibration_restart"
                    no_dumps = len(listdir(self.io.eq_dump_dir))
                    last_step = self.parameters.eq_dump_step * (no_dumps - 1)
                self.parameters.restart_step = last_step
                self.particles.setup(self.parameters, self.species)
                # Restore the original value for future use
                self.parameters.load_method = old_method
                # Update the log file. It is set to the simulation log in the parameters class, but it is correct in the IO class.
                self.parameters.log_file = self.io.log_file
        else:
            self.initialization()

        if self.parameters.plot_style:
            plt.style.use(self.parameters.plot_style)

    def setup_from_dict(self, input_dict: dict):
        """Setup simulations' subclasses from a nested dictionary.

        Parameters
        ----------
        input_dict: dict
            Nested dictionary with all necessary simulations parameters.

        Note
        ----
        This method does the same as:meth:`setup` but without reading a yaml file.
        If you want to update the attributes of this class use :meth:`update_subclasses_from_dict`.
        """

        self.instantiate_subclasses_from_dict(input_dict)

        if self.__name__ == "postprocessing":
            # Create the file paths without creating directories and redefining io attributes
            self.io.make_directory_tree()
            self.io.make_directories()
            self.io.make_files_tree()

            # Read previously stored files
            self.io.read_pickle(self, self.io.directory_tree["simulation"]["path"])
            self.io.copy_params(self.parameters)

            # Print parameters to log file
            if not exists(self.io.log_file):
                # if the file exists do not print the file header
                self.io.file_header()
                self.io.simulation_summary(self)

            self.io.datetime_stamp()

            if self.grab_last_step:
                # Initialize the Particles class attributes by reading the last step
                old_method = self.parameters.load_method
                self.parameters.load_method = "production_restart"
                no_dumps = len(listdir(self.io.prod_dump_dir))
                last_step = self.parameters.prod_dump_step * (no_dumps - 1)
                if no_dumps == 0:
                    self.parameters.load_method = "equilibration_restart"
                    no_dumps = len(listdir(self.io.eq_dump_dir))
                    last_step = self.parameters.eq_dump_step * (no_dumps - 1)
                self.parameters.restart_step = last_step
                self.particles.setup(self.parameters, self.species)
                # Restore the original value for future use
                self.parameters.load_method = old_method
                # Update the log file. It is set to the simulation log in the parameters class, but it is correct in the IO class.
                self.parameters.log_file = self.io.log_file

        else:
            self.initialization()

        if self.parameters.plot_style:
            plt.style.use(self.parameters.plot_style)


class PostProcess(Process):
    """
    Class handling the post-processing stage of a simulation.

    Parameters
    ----------
    input_file : str
        Path to the YAML input file.

    """

    def __init__(self, input_file: str = None, grab_last_step: bool = False):
        self.__name__ = "postprocessing"
        self.grab_last_step = grab_last_step

        super().__init__(input_file)

    def run(self):
        """Calculate all the observables from the YAML input file."""

        if len(self.observables_dict.keys()) == 0:
            print("No observables found in observables_dict")
        else:
            for obs_key, obs_class in self.observables_dict.items():
                obs_class.setup(self.parameters)
                msg = obs_class.pretty_print_msg()
                self.io.write_to_logger(msg)

                if obs_key == "Thermodynamics":
                    # Make Temperature and Energy plots
                    obs_class.temp_energy_plot(self)
                else:
                    obs_class.compute()

        if len(self.transport_dict.keys()) == 0:
            print("No transport coefficients found in tranport_dict")
        else:
            for obs_key, obs_class in self.transport_dict.items():
                obs_class.setup(self.parameters)
                msg = obs_class.pretty_print_msg()
                self.io.write_to_logger(msg)
                obs_class.compute(observable=self.observables_dict[obs_class.required_observable])

    def setup_from_simulation(self, simulation):
        """
        Setup postprocess' subclasses by (shallow) copying them from simulation object.

        Parameters
        ----------
        simulation: :class:`sarkas.core.processes.Simulation`
            Simulation object

        """
        self.parameters = simulation.parameters.__copy__()
        self.integrator = simulation.integrator.__copy__()
        self.potential = simulation.potential.__copy__()
        self.species = simulation.species.copy()
        self.io = simulation.io.__copy__()
        self.io.process = "postprocess"


class PreProcess(Process):
    """
    Wrapper class handling the estimation of time and best parameters of a simulation.

    Parameters
    ----------
    input_file : str
        Path to the YAML input file.

    Attributes
    ----------
    loops: int
        Number of timesteps to run for time and size estimates. Default = 10

    estimate: bool
        Run an estimate for the best PPPM parameters in the simulation. Default=False.

    pm_meshes: numpy.ndarray
        Array of mesh sizes used in the PPPM parameters estimation.

    pp_cells: numpy.ndarray
        Array of simulations box cells used in the PPPM parameters estimation.

    kappa: float
        Screening parameter. Calculated from :meth:`sarkas.potentials.core.Potential.matrix`.

    """

    def __init__(self, input_file: str = None):
        self.__name__ = "preprocessing"
        self.estimate = False
        self.pm_meshes = logspace(3, 7, 12, base=2, dtype=int64)
        # array([16, 24, 32, 48, 56, 64, 72, 88, 96, 112, 128], dtype=int64)
        self.pm_caos = arange(1, 8, dtype=int64)
        self.pp_cells = arange(3, 16, dtype=int64)
        self.kappa = None
        super().__init__(input_file)

    def analytical_approx_pppm(self):
        """Calculate the total force error as given in :cite:`Dharuman2017`."""

        a_min = self.potential.pppm_alpha_ewald * 0.5
        a_max = self.potential.pppm_alpha_ewald * 1.5

        r_min = self.potential.rc * 0.5
        r_max = self.potential.rc * 1.5

        alphas = linspace(a_min, a_max, 101)
        rcuts = linspace(r_min, r_max, 101)

        pm_force_error = zeros(len(alphas))
        pp_force_error = zeros((len(alphas), len(rcuts)))
        total_force_error = zeros((len(alphas), len(rcuts)))

        # TODO: Fix this hack
        potential_copy = self.potential.__copy__()
        potential_copy.setup(self.parameters, self.species)
        # potential_copy = self.potential.__deepcopy__()
        for ia, alpha in enumerate(alphas):
            for ir, rc in enumerate(rcuts):
                potential_copy.rc = rc
                potential_copy.pppm_alpha_ewald = alpha
                tot_err, pm_err, pp_err = force_error_approx_pppm(potential_copy)
                total_force_error[ia, ir] = tot_err
                pm_force_error[ia] = pm_err
                pp_force_error[ia, ir] = pp_err

        return (
            total_force_error,
            pp_force_error,
            pm_force_error,
            rcuts / potential_copy.a_ws,
            alphas * potential_copy.a_ws,
        )

    def green_function_timer(self):
        """Time Potential setup."""

        self.timer.start()
        self.potential.pppm_setup()

        return self.timer.stop()

    def make_color_map(self, rcuts, alphas, chosen_alpha, chosen_rcut, total_force_error):
        """
        Plot a color map of the total force error approximation.

        Parameters
        ----------
        rcuts: numpy.ndarray
            Cut off distances.

        alphas: numpy.ndarray
            Ewald parameters.

        chosen_alpha: float
            Chosen Ewald parameter.

        chosen_rcut: float
            Chosen cut off radius.

        total_force_error: numpy.ndarray
            Force error matrix.

        Raises
        ------
          DeprecationWarning

        """
        warn(
            f"The function has been renamed make_pppm_color_map. make_color_map will be removed in v2.0.0.",
            DeprecationWarning,
        )
        # Line Plot
        self.make_pppm_color_map(rcuts, alphas, chosen_alpha, chosen_rcut, total_force_error)

    @staticmethod
    def make_fit_plot(pp_xdata, pm_xdata, pp_times, pm_times, pp_opt, pm_opt, pp_xlabels, pm_xlabels, fig_path):
        """
        Make a dual plot of the fitted functions.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 7))
        ax[0].plot(pm_xdata, pm_times.mean(axis=-1), "o", label="Measured times")
        # ax[0].plot(pm_xdata, quadratic(pm_xdata, *pm_opt), '--r', label="Fit $f(x) = a + b x + c x^2$")
        ax[1].plot(pp_xdata, pp_times.mean(axis=-1), "o", label="Measured times")
        # ax[1].plot(pp_xdata, linear(pp_xdata, *pp_opt), '--r', label="Fit $f(x) = a x$")

        ax[0].set_xscale("log")
        ax[0].set_yscale("log")

        ax[1].set_xscale("log")
        ax[1].set_yscale("log")

        ax[0].legend()
        ax[1].legend()

        ax[0].set_xticks(pm_xdata)
        ax[0].set_xticklabels(pm_xlabels)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ax[1].set_xticks(pp_xdata[0:-1:3])
        ax[1].set_xticklabels(pp_xlabels)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ax[0].set_title("PM calculation")
        ax[1].set_title("PP calculation")

        ax[0].set_xlabel("Mesh sizes")
        ax[1].set_xlabel(r"$r_c / a_{ws}$")
        fig.tight_layout()
        fig.savefig(join(fig_path, "Timing_Fit.png"))

    def make_line_plot(self, rcuts, alphas, chosen_alpha, chosen_rcut, total_force_error):
        """
        Plot selected values of the total force error approximation.

        Parameters
        ----------
        rcuts: numpy.ndarray
            Cut off distances.

        alphas: numpy.ndarray
            Ewald parameters.

        chosen_alpha: float
            Chosen Ewald parameter.

        chosen_rcut: float
            Chosen cut off radius.

        total_force_error: numpy.ndarray
            Force error matrix.

        Raises
        ------
            : DeprecationWarning
        """
        warn(
            f"The function has been renamed make_pppm_line_plot. make_line_plot will be removed in v2.0.0.",
            DeprecationWarning,
        )
        # Line Plot
        self.make_pppm_line_plot(rcuts, alphas, chosen_alpha, chosen_rcut, total_force_error)

    def make_lagrangian_plot(self):
        "TODO: complete this."
        c_mesh, m_mesh = meshgrid(self.pp_cells, self.pm_meshes)
        fig = plt.figure()
        ax = fig.add_subplot(111)  # projection='3d')
        # CS = ax.plot_surface(m_mesh, c_mesh, self.lagrangian, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        CS = ax.contourf(
            m_mesh, c_mesh, self.lagrangian, norm=LogNorm(vmin=self.lagrangian.min(), vmax=self.lagrangian.max())
        )
        CS2 = ax.contour(CS, colors="w")
        ax.clabel(CS2, fmt="%1.0e", colors="w")
        fig.colorbar(CS)
        ax.scatter(self.best_mesh, self.best_cells, s=200, c="k")
        ax.set_xlabel("Mesh size")
        ax.set_ylabel(r"Cells = $L/r_c$")
        ax.set_title("2D Lagrangian")
        fig.savefig(join(self.io.preprocessing_dir, "2D_Lagrangian.png"))

        # ax[1].set_xticks([8, 16, 32, 64, 128])
        # ax[1].set_xticklabels([8, 16, 32, 64, 128])

    def make_pppm_line_plot(self, rcuts, alphas, chosen_alpha, chosen_rcut, total_force_error):
        """
        Plot selected values of the total force error approximation.

        Parameters
        ----------
        rcuts: numpy.ndarray
            Cut off distances.

        alphas: numpy.ndarray
            Ewald parameters.

        chosen_alpha: float
            Chosen Ewald parameter.

        chosen_rcut: float
            Chosen cut off radius.

        total_force_error: numpy.ndarray
            Force error matrix.

        """
        # Plot the results
        fig_path = self.pppm_plots_dir

        fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(19, 7))
        linestyles = [(0, (5, 10)), "dashed", "solid", "dashdot", (0, (3, 10, 1, 10))]
        indexes = [30, 40, 50, 60, 70]
        for lns, i in zip(linestyles, indexes):
            min_rc = rcuts[total_force_error[i, :].argmin()]
            rc_lbl = (
                r"$\alpha a_{ws} = "
                + "{:.2f}$".format(alphas[i])
                + r" min @ $r_c = "
                + "{:.2f}".format(min_rc)
                + r" a_{\rm ws}$"
            )
            ax[0].plot(rcuts, total_force_error[i, :], ls=lns, label=rc_lbl)
            min_a = alphas[total_force_error[:, i].argmin()]
            a_lbl = (
                r"$r_c = {:.2f}".format(rcuts[i])
                + " a_{ws}$"
                + r" min @ $\alpha_{\rm min} a_{ws} = "
                + "{:.2f}$".format(min_a)
            )
            ax[1].plot(alphas, total_force_error[:, i], ls=lns, label=a_lbl)

        ax[0].set(ylabel=r"$\Delta F^{approx}_{tot}$", xlabel=r"$r_c/a_{ws}$", yscale="log")
        ax[1].set(xlabel=r"$\alpha \; a_{ws}$", yscale="log")

        ax[0].axvline(chosen_rcut, ls="--", c="k")
        ax[0].axhline(self.potential.force_error, ls="--", c="k")
        ax[1].axhline(self.potential.force_error, ls="--", c="k")
        ax[1].axvline(chosen_alpha, ls="--", c="k")

        if rcuts[-1] * self.parameters.a_ws > 0.5 * self.parameters.box_lengths.min():
            ax[0].axvline(0.5 * self.parameters.box_lengths.min() / self.parameters.a_ws, c="r", label=r"$L/2$")

        for a in ax:
            a.grid(True, alpha=0.3)
            a.legend(loc="best")

        fig.suptitle(
            r"Parameters  $N = {}, \quad M = {}, \quad p = {}, \quad \kappa = {:.2f}$".format(
                self.parameters.total_num_ptcls,
                self.potential.pppm_mesh[0],
                self.potential.pppm_cao[0],
                self.parameters.a_ws / self.potential.screening_length,
            )
        )
        fig.savefig(join(fig_path, "LinePlot_ForceError_" + self.io.job_id + ".png"))

    def make_pppm_color_map(self, rcuts, alphas, chosen_alpha, chosen_rcut, total_force_error):
        """
        Plot a color map of the total force error approximation.

        Parameters
        ----------
        rcuts: numpy.ndarray
            Cut off distances.

        alphas: numpy.ndarray
            Ewald parameters.

        chosen_alpha: float
            Chosen Ewald parameter.

        chosen_rcut: float
            Chosen cut off radius.

        total_force_error: numpy.ndarray
            Force error matrix.
        """
        # Plot the results
        fig_path = self.pppm_plots_dir

        r_mesh, a_mesh = meshgrid(rcuts, alphas)
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        # if total_force_error.min() == 0.0:
        #     minv = 1e-120
        # else:
        #     minv = total_force_error.min()
        # total_force_error[total_force_error == 0.0] = minv
        CS = ax.pcolormesh(a_mesh, r_mesh, total_force_error, shading="auto", norm=LogNorm())
        CS2 = ax.contour(a_mesh, r_mesh, total_force_error, levels=10, colors="w", norm=LogNorm())
        ax.clabel(CS2, fmt="%1.0e", colors="w")
        ax.scatter(chosen_alpha, chosen_rcut, s=200, c="k")
        if rcuts[-1] * self.parameters.a_ws > 0.5 * self.parameters.box_lengths.min():
            ax.axhline(0.5 * self.parameters.box_lengths.min() / self.parameters.a_ws, c="r", label=r"$L/2$")
        # ax.tick_parameters(labelsize=fsz)
        ax.set_xlabel(r"$\alpha \;a_{ws}$")
        ax.set_ylabel(r"$r_c/a_{ws}$")
        ax.set_title(
            r"Parameters  $N = {}, \quad M = {}, \quad p = {}, \quad \kappa = {:.2f}$".format(
                self.parameters.total_num_ptcls,
                self.potential.pppm_mesh[0],
                self.potential.pppm_cao[0],
                self.parameters.a_ws / self.potential.screening_length,
            )
        )
        clb = fig.colorbar(CS)
        clb.set_label(r"$\Delta F^{approx}_{tot}(r_c,\alpha)$", va="bottom", rotation=270)
        fig.tight_layout()
        fig.savefig(join(fig_path, "ClrMap_ForceError_" + self.io.job_id + ".png"))

    def make_timing_plots(self, data_df: DataFrame = None):
        """
        Makes a figure with three subplots of the CPU times vs PPPM parameters.\n
        The first plot is PP acc time vs the number of cells at different Mesh sizes.\n
        The second plot is the PM acc time vs the mesh size at different charge assignment orders.\n
        The third plot is the time for the calculation of the optimal green's function
        for different charge asssignment orders.

        Parameters
        ----------

        data_df : pandas.DataFrame, Optional
            Timing study data. If `None` it will look for previously saved data, otherwise it will run
            :meth:`sarkas.processes.PreProcess.timing_study_calculation` to calculate the data. Default is `None`.

        """

        if not data_df:
            try:
                data_df = read_csv(
                    join(self.io.preprocessing_dir, f"TimingStudy_data_{self.io.job_id}.csv"), index_col=False
                )
                self.dataframe = data_df.copy()
            except FileNotFoundError:
                print(f"I could not find the data from the timing study. Running the timing study now.")
                self.timing_study_calculation()
        else:
            data_df = self.dataframe.copy(deep=True)

        fig, ax = plt.subplots(1, 3, figsize=(21, 7))
        scatterplot(data=data_df, x="pp_cells", y="pp_acc_time [s]", hue="M_x", s=100, palette="viridis", ax=ax[0])

        scatterplot(data=data_df, x="M_x", y="pm_acc_time [s]", hue="pppm_cao_x", s=150, palette="viridis", ax=ax[1])

        scatterplot(data=data_df, x="M_x", y="G_k time [s]", hue="pppm_cao_x", s=150, palette="viridis", ax=ax[2])
        # ax[0].legend(ncol = 2)
        ax[0].set(yscale="log", xlabel="LCL Cells", ylabel="PP Time [s]")
        ax[1].set(yscale="log", xlabel="Mesh", ylabel="PM Time [s]")
        ax[2].set(yscale="log", xlabel="Mesh", ylabel="Green Function Time [s]")
        ax[1].set_xscale("log", base=2)
        ax[2].set_xscale("log", base=2)
        fig_path = self.pppm_plots_dir
        fig.savefig(join(fig_path, f"PPPM_Times_{self.io.job_id}.png"))

        print(f"\nFigures can be found in {self.pppm_plots_dir}")

    def make_force_v_timing_plot(self, data_df: DataFrame = None):
        """Make contour maps of the force error and total acc time as functions of LCL cells and PM meshes for each
        charge assignment order sequence.

        Parameters
        ----------
        data_df : pandas.DataFrame, Optional
            Timing study data. If `None` it will look for previously saved data, otherwise it will run
            :meth:`sarkas.processes.PreProcess.timing_study_calculation` to calculate the data. Default is `None`.

        """
        from scipy.interpolate import griddata

        fig_path = self.pppm_plots_dir
        if not data_df:
            try:
                data_df = read_csv(
                    join(self.io.preprocessing_dir, f"TimingStudy_data_{self.io.job_id}.csv"), index_col=False
                )
                self.dataframe = data_df.copy()
            except FileNotFoundError:
                print(f"I could not find the data from the timing study. Running the timing study now.")
                self.timing_study_calculation()
        else:
            data_df = self.dataframe.copy(deep=True)

        # Plot the results
        for _, cao in enumerate(self.pm_caos):
            mask = self.dataframe["pppm_cao_x"] == cao
            df = data_df[mask][
                ["M_x", "pp_cells", "force error [measured]", "pp_acc_time [s]", "pm_acc_time [s]", "tot_acc_time [s]"]
            ]

            # 2D-arrays from DataFrame
            n_meshes = len(df["M_x"].unique())
            x1 = logspace(log2(df["M_x"].min()), log2(df["M_x"].max()), 5 * n_meshes, base=2)
            n_cells = len(df["pp_cells"].unique())
            y1 = linspace(df["pp_cells"].min(), df["pp_cells"].max(), 5 * n_cells)

            m_mesh, c_mesh = meshgrid(x1, y1)

            # Interpolate unstructured D-dimensional data.
            tot_time_map = griddata((df["M_x"], df["pp_cells"]), df["tot_acc_time [s]"], (m_mesh, c_mesh))
            force_error_map = griddata((df["M_x"], df["pp_cells"]), df["force error [measured]"], (m_mesh, c_mesh))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
            if force_error_map.min() == 0.0:
                minv = 1e-120
            else:
                minv = force_error_map.min()

            maxt = force_error_map.max()
            nlvl = 12
            lvls = logspace(log10(minv), log10(maxt), nlvl)

            luxmap = get_cmap("viridis", nlvl)
            luxnorm = LogNorm(vmin=minv, vmax=maxt)
            CS = ax1.contourf(m_mesh, c_mesh, force_error_map, levels=lvls, cmap=luxmap, norm=luxnorm)
            clb = fig.colorbar(ScalarMappable(norm=luxnorm, cmap=luxmap), ax=ax1)
            clb.set_label(r"Force Error  [$Q^2/ a_{\rm ws}^2$] " + f"@ cao = {cao}", rotation=270, va="bottom")
            CS2 = ax1.contour(CS, colors="w")
            ax1.clabel(CS2, fmt="%1.0e", colors="w")

            if cao == self.potential.pppm_cao[0]:
                input_Nc = int(self.potential.box_lengths[0] / self.potential.rc)
                ax1.scatter(self.potential.pppm_mesh[0], input_Nc, s=200, c="k")

            ax1.set_xscale("log", base=2)
            ax1.set(xlabel="Mesh size", ylabel=r"LCL Cells", title=f"Force Error Map @ cao = {cao}")

            # Timing Plot
            maxt = tot_time_map.max()
            mint = tot_time_map.min()
            # nlvl = 13
            lvls = logspace(log10(mint), log10(maxt), nlvl)
            luxmap = get_cmap("viridis", nlvl)
            luxnorm = LogNorm(vmin=minv, vmax=maxt)

            CS = ax2.contourf(m_mesh, c_mesh, tot_time_map, levels=lvls, cmap=luxmap)
            CS2 = ax2.contour(CS, colors="w", levels=lvls)
            ax2.clabel(CS2, fmt="%.2e", colors="w")
            # fig.colorbar(, ax = ax2)
            clb = fig.colorbar(ScalarMappable(norm=luxnorm, cmap=luxmap), ax=ax2)
            clb.set_label("CPU Time [s]", rotation=270, va="bottom")
            if cao == self.potential.pppm_cao[0]:
                input_Nc = int(self.potential.box_lengths[0] / self.potential.rc)
                ax2.scatter(self.potential.pppm_mesh[0], input_Nc, s=200, c="k")

            ax2.set_xscale("log", base=2)
            ax2.set(xlabel="Mesh size", title=f"Timing Map @ cao = {cao}")
            fig.savefig(join(fig_path, f"ForceErrorMap_v_Timing_cao_{cao}_{self.io.job_id}.png"))

    def postproc_estimates(self):
        # POST- PROCESSING
        self.io.postprocess_info(self, observable="header")
        # Header of process
        process_title = f"{'PostProcessing':^80}"
        msg = f"{'':*^80}\n {process_title} \n{'':*^80}"
        print_to_logger(msg, self.parameters.log_file, self.parameters.verbose)

        for _, obs_class in self.observables_dict.items():
            obs_class.setup(self.parameters)
            msg += self.rdf.pretty_print_msg()

        print_to_logger(msg, self.parameters.log_file, self.parameters.verbose)

    def make_pppm_plots_dir(self):
        self.pppm_plots_dir = join(self.io.directory_tree["preprocessing"]["path"], "PPPM_Plots")

        if not exists(self.pppm_plots_dir):
            mkdir(self.pppm_plots_dir)

    def pppm_approximation(self):
        """
        Calculate the force error for a PPPM simulation using analytical approximations.\n
        Plot the force error in the parameter space.
        """

        self.make_pppm_plots_dir()

        # Calculate Force error from analytic approximation given in Dharuman et al. J Chem Phys 2017
        total_force_error, pp_force_error, pm_force_error, rcuts, alphas = self.analytical_approx_pppm()
        chosen_alpha = self.potential.pppm_alpha_ewald * self.parameters.a_ws
        chosen_rcut = self.potential.rc / self.parameters.a_ws

        # mesh_dir = join(self.pppm_plots_dir, 'Mesh_{}'.format(self.potential.pppm_mesh[0]))
        # if not exists(mesh_dir):
        #     mkdir(mesh_dir)
        #
        # cell_num = int(self.parameters.box_lengths.min() / self.potential.rc)
        # cell_dir = join(mesh_dir, 'Cells_{}'.format(cell_num))
        # if not exists(cell_dir):
        #     mkdir(cell_dir)
        #
        # self.pppm_plots_dir = cell_dir

        # Color Map
        self.make_pppm_color_map(rcuts, alphas, chosen_alpha, chosen_rcut, total_force_error)

        # Line Plot
        self.make_pppm_line_plot(rcuts, alphas, chosen_alpha, chosen_rcut, total_force_error)

        msg = f"\nFigures can be found in {self.pppm_plots_dir}"
        self.io.write_to_logger(msg)

    def remove_preproc_dumps(self):
        # Delete the energy files created during the estimation runs
        os_remove(self.io.eq_energy_filename)
        os_remove(self.io.prod_energy_filename)

        # Delete dumps created during the estimation runs
        for npz in listdir(self.io.eq_dump_dir):
            os_remove(join(self.io.eq_dump_dir, npz))

        for npz in listdir(self.io.prod_dump_dir):
            os_remove(join(self.io.prod_dump_dir, npz))

        if self.parameters.magnetized and self.parameters.electrostatic_equilibration:
            os_remove(self.io.mag_energy_filename)
            # Remove dumps
            for npz in listdir(self.io.mag_dump_dir):
                os_remove(join(self.io.mag_dump_dir, npz))

    def run(
        self,
        loops: int = 10,
        timing: bool = True,
        timing_study: bool = False,
        pppm_estimate: bool = False,
        postprocessing: bool = False,
        remove: bool = False,
    ):
        """
        Estimate the time of the simulation and best parameters if wanted.

        Parameters
        ----------
        loops : int
            Number of loops over which to average the acceleration calculation. Default = 10.

        timing : bool
            Flag for estimating simulation times. Default =True.

        timing_study : bool
            Flag for estimating time for simulation parameters.

        pppm_estimate : bool
            Flag for showing the force error plots in case of pppm algorithm.

        postprocessing : bool
            Flag for calculating Post processing parameters.

        remove : bool
            Flag for removing energy files and dumps created during times estimation. Default = False.

        """

        # Clean everything
        plt.close("all")

        # Set the screening parameter
        self.kappa = self.potential.matrix[0, 0, 1] if self.potential.type == "yukawa" else 0.0

        if timing:
            self.time_n_space_estimates(loops=loops)

        if remove:
            self.remove_preproc_dumps()

        if pppm_estimate:
            self.pppm_approximation()

        if timing_study:
            self.timing_study_calculation()
            self.make_timing_plots()
            self.make_force_v_timing_plot()
            print(f"\nFigures can be found in {self.pppm_plots_dir}")

        if postprocessing:
            self.postproc_estimates()

    def time_acceleration(self, loops: int = 11):
        """
        Run loops number of acceleration calculations for timing estimate.


        Parameters
        ----------
        loops: int
            Number of simulation steps to run. Default = 11.

        """

        if self.potential.linked_list_on:
            self.pp_acc_time = zeros(loops)
            for i in trange(loops, desc="PP acceleration timer", disable=not self.parameters.verbose):
                self.timer.start()
                self.potential.update_linked_list(self.particles)
                self.pp_acc_time[i] = self.timer.stop()

            # Calculate the mean excluding the first value because that time include numba compilation time
            pp_mean_time = self.timer.time_division(self.pp_acc_time[1:].mean())

            self.io.preprocess_timing("PP", pp_mean_time, loops)

        # PM acceleration
        if self.potential.pppm_on:
            self.pm_acc_time = zeros(loops)
            for i in trange(loops, desc="PM acceleration timer", disable=not self.parameters.verbose):
                self.timer.start()
                self.potential.update_pm(self.particles)
                self.pm_acc_time[i] = self.timer.stop()
            pm_mean_time = self.timer.time_division(self.pm_acc_time[1:].mean())
            self.io.preprocess_timing("PM", pm_mean_time, loops)

        if self.potential.method == "fmm":
            self.fmm_acc_time = zeros(loops)

            for i in range(loops):
                self.timer.start()
                self.integrator.update_accelerations(self.particles)
                self.fmm_acc_time[i] = self.timer.stop()
            fmm_mean_time = self.timer.time_division(self.fmm_acc_time[:].mean())
            self.io.preprocess_timing("FMM", fmm_mean_time, loops)

    def time_evolution_loop(self, loops: int = 11):
        """Run several loops of the equilibration and production phase to estimate the total time of the simulation.

        Parameters
        ----------
        loops: int
            Number of simulation steps to run. Default = 11.

        """

        msg = f"\nRunning {loops} steps for each phase to estimate simulation times\n"
        self.io.write_to_logger(msg)

        # Run few equilibration steps to estimate the equilibration time
        if self.parameters.equilibration_phase and self.parameters.electrostatic_equilibration:
            self.integrator.update = self.integrator.type_setup(self.integrator.equilibration_type)
            self.timer.start()
            self.evolve("equilibration", self.integrator.thermalization, 0, loops, self.parameters.eq_dump_step)
            self.eq_mean_time = self.timer.stop() / loops
            # Print the average equilibration & production times
            self.io.preprocess_timing("Equilibration", self.timer.time_division(self.eq_mean_time), loops)

        if self.parameters.magnetized and self.parameters.electrostatic_equilibration:
            self.integrator.update = self.integrator.type_setup(self.integrator.magnetization_type)
            self.timer.start()
            self.evolve("magnetization", self.integrator.thermalization, 0, loops, self.parameters.mag_dump_step)
            self.mag_mean_time = self.timer.stop() / loops
            # Print the average equilibration & production times
            self.io.preprocess_timing("Magnetization", self.timer.time_division(self.mag_mean_time), loops)

        # Run few production steps to estimate the equilibration time
        self.integrator.update = self.integrator.type_setup(self.integrator.production_type)
        self.timer.start()
        self.evolve("production", False, 0, loops, self.parameters.prod_dump_step)
        self.prod_mean_time = self.timer.stop() / loops
        self.io.preprocess_timing("Production", self.timer.time_division(self.prod_mean_time), loops)

        if self.parameters.equilibration_phase and self.parameters.electrostatic_equilibration:
            # Print the estimate for the full run
            eq_prediction = self.eq_mean_time * self.parameters.equilibration_steps
            self.io.time_stamp("Equilibration", self.timer.time_division(eq_prediction))
        else:
            eq_prediction = 0.0

        if self.parameters.magnetized and self.parameters.electrostatic_equilibration:
            mag_prediction = self.mag_mean_time * self.parameters.magnetization_steps
            self.io.time_stamp("Magnetization", self.timer.time_division(mag_prediction))
            eq_prediction += mag_prediction

        prod_prediction = self.prod_mean_time * self.parameters.production_steps
        self.io.time_stamp("Production", self.timer.time_division(prod_prediction))

        tot_time = eq_prediction + prod_prediction
        self.io.time_stamp("Total Run", self.timer.time_division(tot_time))

    def time_n_space_estimates(self, loops: int = 10):
        """Estimate simulation times and space

        Parameters
        ----------
        loops: int
            Number of simulation steps to run. Default = 10.

        """

        if loops:
            loops += 1

        self.io.preprocess_timing("header", [0, 0, 0, 0, 0, 0], 0)
        if self.potential.pppm_on:
            green_time = self.timer.time_division(self.green_function_timer())
            self.io.preprocess_timing("GF", green_time, 0)

        self.time_acceleration(loops)

        self.time_evolution_loop(loops)

        self.directory_sizes()

    def timing_study_calculation(self):
        """Estimate the best number of mesh points and cutoff radius."""

        self.pppm_plots_dir = join(self.io.directory_tree["preprocessing"]["path"], "PPPM_Plots")
        if not exists(self.pppm_plots_dir):
            mkdir(self.pppm_plots_dir)

        msg = "\n\n{:=^70} \n".format(" Timing Study ")
        self.io.write_to_logger(msg)

        self.input_rc = self.potential.rc
        self.input_mesh = self.potential.pppm_mesh.copy()
        self.input_alpha = self.potential.pppm_alpha_ewald
        self.input_cao = self.potential.pppm_cao.copy()

        data = DataFrame()
        # Rescaling constant to calculate the PP force error
        rescaling_constant = (
            sqrt(self.potential.total_num_ptcls) * self.potential.a_ws**2 / sqrt(self.potential.pbox_volume)
        )

        # Set the maximum number of cells to be L / (2 * a_ws). 2* a_ws is the closest two particles can be.
        max_cells = int(0.5 * self.parameters.box_lengths.min() / self.parameters.a_ws)
        if max_cells != self.pp_cells[-1]:
            self.pp_cells = arange(3, max_cells, dtype=int)

        # Start the loop for averaging PM and PP acceleration times
        for _, m in enumerate(
            tqdm(self.pm_meshes, desc="Looping over the PM meshes", disable=not self.parameters.verbose)
        ):
            # Setup PM params
            self.potential.pppm_mesh = m * array([1, 1, 1], dtype=int)
            self.potential.pppm_alpha_ewald = 0.3 * m / self.potential.box_lengths.min()
            self.potential.pppm_h_array = self.potential.box_lengths / self.potential.pppm_mesh
            self.io.write_to_logger(f"\n Mesh = {m, m, m}:\n\t cao = ")

            for _, cao in enumerate(self.pm_caos):
                self.io.write_to_logger(f"{cao}, ")
                self.potential.pppm_cao = cao * array([1, 1, 1], dtype=int)

                # Update the potential matrix since alpha has changed
                if self.potential.type == "qsp":
                    self.potential.pot_update_params(self.potential, self.species)
                else:
                    self.potential.pot_update_params(self.potential, self.species)
                # The Green's function depends on alpha, Mesh and cao. It also updates the pppm_pm_err
                green_time = self.green_function_timer()

                # Calculate the PM acceleration timing 3x and average
                pm_acc_time = 0.0
                for it in range(3):
                    self.timer.start()
                    self.potential.update_pm(self.particles)
                    pm_acc_time += self.timer.stop() / 3.0

                # Loop over the number of cells
                for _, cell in enumerate(self.pp_cells):
                    # Cutoff radius is the side of the cells.
                    self.potential.rc = self.potential.box_lengths.min() / cell

                    # Update the potential pp error
                    self.potential.pppm_pp_err = force_error_analytic_pp(
                        self.potential.type,
                        self.potential.rc,
                        self.potential.screening_length,
                        self.potential.pppm_alpha_ewald,
                        rescaling_constant,
                    )

                    # Note: the PM error does not depend on rc. Only on alpha and it is given by G_k
                    self.potential.force_error = sqrt(self.potential.pppm_pp_err**2 + self.potential.pppm_pm_err**2)

                    # The PP acceleration does not depend on cao.
                    # However, it still needs to be in its loop for updating the dataframe.
                    pp_acc_time = 0.0
                    for it in range(3):
                        self.timer.start()
                        self.potential.update_linked_list(self.particles)
                        pp_acc_time += self.timer.stop() / 3.0

                    data = data.append(
                        {
                            "pp_cells": cell,
                            "r_cut": self.potential.rc,
                            "pppm_alpha_ewald": self.potential.pppm_alpha_ewald,
                            "pppm_cao_x": self.potential.pppm_cao[0],
                            "pppm_cao_y": self.potential.pppm_cao[1],
                            "pppm_cao_z": self.potential.pppm_cao[2],
                            "M_x": self.potential.pppm_mesh[0],
                            "M_y": self.potential.pppm_mesh[1],
                            "M_z": self.potential.pppm_mesh[2],
                            "Mesh volume": self.potential.pppm_mesh.prod(),
                            "Mesh": f"{self.potential.pppm_mesh[0], self.potential.pppm_mesh[1], self.potential.pppm_mesh[2]}",
                            "h_x": self.potential.pppm_h_array[0],
                            "h_y": self.potential.pppm_h_array[1],
                            "h_z": self.potential.pppm_h_array[2],
                            "h_M volume": self.potential.pppm_h_array.prod(),
                            "h_x alpha": self.potential.pppm_h_array[0] * self.potential.pppm_alpha_ewald,
                            "h_y alpha": self.potential.pppm_h_array[1] * self.potential.pppm_alpha_ewald,
                            "h_z alpha": self.potential.pppm_h_array[2] * self.potential.pppm_alpha_ewald,
                            "h_M a_ws^3": self.potential.pppm_h_array.prod() * self.potential.pppm_alpha_ewald**3,
                            "G_k time [s]": green_time * 1.0e-9,
                            "pp_acc_time [s]": pp_acc_time * 1.0e-9,
                            "pm_acc_time [s]": pm_acc_time * 1.0e-9,
                            "tot_acc_time [s]": (pp_acc_time + pm_acc_time) * 1.0e-9,
                            "pppm_pp_error [measured]": self.potential.pppm_pp_err,
                            "pppm_pm_error [measured]": self.potential.pppm_pm_err,
                            "force error [measured]": self.potential.force_error,
                        },
                        ignore_index=True,
                    )

        self.dataframe = data
        csv_location = join(self.io.preprocessing_dir, f"TimingStudy_data_{self.io.job_id}.csv")
        self.dataframe.to_csv(csv_location, index=False)

        # Reset the original values.
        self.potential.rc = self.input_rc
        self.potential.pppm_mesh = self.input_mesh.copy()
        self.potential.pppm_alpha_ewald = self.input_alpha
        self.potential.pppm_cao = self.input_cao.copy()
        self.potential.setup(self.parameters, self.species)

        msg = (
            f"\nThe force error and computation times can be found in a dataframe at PreProcess.dataframe "
            f"and the corresponding csv file is saved in {csv_location}"
        )
        self.io.write_to_logger(msg)

        # pm_popt = zeros((len(self.pm_caos), 2))
        # for ic, cao in enumerate(self.pm_caos):
        #     # Fit the PM times
        #     pm_popt[ic, :], _ = curve_fit(
        #         lambda x, a, b: a + 5 * b * x ** 3 * log2(x ** 3), self.pm_meshes, pm_times[:, ic]
        #     )
        #     fit_str = (
        #         r"Fit = $a_2 + 5 a_3 M^3 \log_2(M^3)$  [s]"
        #         + "\n"
        #         + r"$a_2 = ${:.4e}, $a_3 = ${:.4e} ".format(pm_popt[ic, 0], pm_popt[ic, 1])
        #     )
        #     print(f"\nPM Time for cao {cao}: " + fit_str)
        #
        #     # Fit the PP Times
        #     pp_popt, _ = curve_fit(
        #         lambda x, a, b: a + b / x ** 3,
        #         self.pp_cells,
        #         pp_times.mean(axis=0),
        #         p0=[pp_times.mean(axis=0)[0], self.parameters.total_num_ptcls],
        #         bounds=(0, [pp_times.mean(axis=0)[0], 1e9]),
        #     )
        #     fit_pp_str = (
        #         r"Fit = $a_0 + a_1 / N_c^3$  [s]"
        #         + "\n"
        #         + "$a_0 = ${:.4e},  $a_1 = ${:.4e}".format(pp_popt[0], pp_popt[1])
        #     )
        #     print(f"\nPP Time cao {cao}:" + fit_pp_str)
        #
        #     # Start the plot
        #     fig, (ax_pp, ax_pm) = plt.subplots(1, 2, sharey=True, figsize=(12, 7))
        #     ax_pm.plot(self.pm_meshes, pm_times[:, ic], "o", label=f"Measured cao: {cao}")
        #     ax_pm.plot(
        #         self.pm_meshes,
        #         pm_popt[ic, 0] + 5 * pm_popt[ic, 1] * self.pm_meshes ** 3 * log2(self.pm_meshes ** 3),
        #         ls="--",
        #         label="Fit",
        #     )
        #     ax_pm.annotate(
        #         text=fit_str,
        #         xy=(self.pm_meshes[-1], pm_times[-1, ic]),
        #         xytext=(self.pm_meshes[0], pm_times[-1, ic]),
        #         bbox=dict(boxstyle="round4", fc="white", ec="k", lw=2),
        #     )
        #
        #     ax_pm.set(title=f"PM calculation time and estimate @ cao = {cao}", yscale="log", xlabel="Mesh size")
        #     ax_pm.set_xscale("log", base=2)
        #     ax_pm.legend(ncol=2)
        #
        #     # Scatter Plot the PP Times
        #     self.tot_time_map = zeros(pp_times.shape)
        #     for j, mesh_points in enumerate(self.pm_meshes):
        #         self.tot_time_map[j, :] = pm_times[j, ic] + pp_times[j, :]
        #         ax_pp.plot(self.pp_cells, pp_times[j], "o", label=r"@ Mesh {}$^3$".format(mesh_points))
        #
        #     # Plot the Fit PP times
        #     ax_pp.plot(self.pp_cells, pp_popt[0] + pp_popt[1] / self.pp_cells ** 3, ls="--", label="Fit")
        #     ax_pp.legend(ncol=2)
        #     ax_pp.annotate(
        #         text=fit_pp_str,
        #         xy=(self.pp_cells[0], pp_times[0, 0]),
        #         xytext=(self.pp_cells[0], pp_times[-1, -1]),
        #         bbox=dict(boxstyle="round4", fc="white", ec="k", lw=2),
        #     )
        #     ax_pp.set(
        #         title=f"PP calculation time and estimate @ cao = {cao}",
        #         yscale="log",
        #         ylabel="CPU Times [s]",
        #         xlabel=r"$N_c $ = Cells",
        #     )
        #     fig.tight_layout()
        #     fig.savefig(join(self.pppm_plots_dir, f"Times_cao_{cao}_" + self.io.job_id + ".png"))
        #
        #     self.make_force_v_timing_plot(ic)
        # self.lagrangian = np.empty((len(self.pm_meshes), len(self.pp_cells)))
        # self.tot_times = np.empty((len(self.pm_meshes), len(self.pp_cells)))
        # self.pp_times = pp_times.copy()
        # self.pm_times = pm_times.copy()
        # for i in range(len(self.pm_meshes)):
        #     self.tot_times[i, :] = pp_times[i] + pm_times[i]
        #     self.lagrangian[i, :] = self.force_error_map[i, :]
        #
        # best = np.unravel_index(self.lagrangian.argmin(), self.lagrangian.shape)
        # self.best_mesh = self.pm_meshes[best[0]]
        # self.best_cells = self.pp_cells[best[1]]

        # self.make_lagrangian_plot()
        #
        # # set the best parameter
        # self.potential.pppm_mesh = self.best_mesh * np.ones(3, dtype=int)
        # self.potential.rc = self.parameters.box_lengths.min() / self.best_cells
        # self.potential.pppm_alpha_ewald = 0.3 * self.best_mesh / self.parameters.box_lengths.min()
        # self.potential.pppm_setup(self.parameters)
        #
        # # print report
        # self.io.timing_study(self)
        # # time prediction
        # self.predicted_times = pp_times[best] + pm_times[best[0]]
        # # Print estimate of run times
        # self.io.time_stamp('Equilibration',
        #                    self.timer.time_division(self.predicted_times * self.parameters.equilibration_steps))
        # self.io.time_stamp('Production',
        #                    self.timer.time_division(self.predicted_times * self.parameters.production_steps))
        # self.io.time_stamp('Total Run',
        #                    self.timer.time_division(self.predicted_times * (self.parameters.equilibration_steps
        #                                                                     + self.parameters.production_steps)))


class Simulation(Process):
    """
    Sarkas simulation wrapper. This class manages the entire simulation and its small moving parts.

    Parameters
    ----------
    input_file : str
        Path to the YAML input file.

    """

    def __init__(self, input_file: str = None):
        self.__name__ = "simulation"
        super().__init__(input_file)

    def check_restart(self, phase):
        """
        Check if the simulation is a restart.

        Parameters
        ----------
        phase: str
            Simulation phase, e.g. equilibration, Magnetization, production.

        Returns
        -------
        it_start: int
            Restart step.

        """

        if self.parameters.verbose:
            print(f"\n\n{phase.capitalize():-^70} \n")

        # Check if this is restart
        if self.parameters.load_method[:2] == phase[:2] and self.parameters.load_method[-7:] == "restart":
            it_start = self.parameters.restart_step
        else:
            it_start = 0
            self.io.dump(phase, self.particles, 0)
        return it_start

    def equilibrate(self):
        """
        Run the time integrator with the thermostat to evolve the system to its thermodynamics equilibrium state.
        """

        it_start = self.check_restart(phase="equilibration")
        self.integrator.update = self.integrator.type_setup(self.integrator.equilibration_type)
        # Start timer, equilibrate, and print run time.
        self.timer.start()
        self.evolve(
            "equilibration",
            self.integrator.thermalization,
            it_start,
            self.parameters.equilibration_steps,
            self.parameters.eq_dump_step,
        )
        time_eq = self.timer.stop()
        self.io.time_stamp("Equilibration", self.timer.time_division(time_eq))

    # def check_equil_dist(self):
    #
    #     vel_dist = VelocityDistribution()
    #     vel_dist.setup(
    #         params=self.parameters,
    #         phase="equilibration",
    #         no_slices=1,
    #         max_no_moment=6,
    #         multi_run_average=False,
    #         dimensional_average=False,
    #         runs=1,
    #     )
    #     vel_dist.compute()
    #
    #     pass

    def magnetize(self):
        # Check for magnetization phase
        it_start = self.check_restart(phase="magnetization")
        # Update integrator
        self.integrator.update = self.integrator.type_setup(self.integrator.magnetization_type)
        # Start timer, magnetize, and print run time.
        self.timer.start()
        self.evolve(
            "magnetization",
            self.integrator.thermalization,
            it_start,
            self.parameters.magnetization_steps,
            self.parameters.mag_dump_step,
        )
        time_eq = self.timer.stop()
        self.io.time_stamp("Magnetization", self.timer.time_division(time_eq))

    def produce(self):
        it_start = self.check_restart(phase="production")

        self.integrator.update = self.integrator.type_setup(self.integrator.production_type)
        # Update measurement flag for rdf.
        self.potential.measure = True
        self.timer.start()
        self.evolve("production", False, it_start, self.parameters.production_steps, self.parameters.prod_dump_step)
        time_eq = self.timer.stop()
        self.io.time_stamp("Production", self.timer.time_division(time_eq))

    def run(self):
        """Run the simulation."""
        time0 = self.timer.current()

        if self.parameters.equilibration_phase and self.parameters.electrostatic_equilibration:
            if self.parameters.remove_initial_drift:
                self.particles.remove_drift()
            self.equilibrate()

        if self.parameters.magnetization_phase:
            if self.parameters.remove_initial_drift:
                self.particles.remove_drift()
            self.magnetize()

        if self.parameters.production_phase:
            if self.parameters.remove_initial_drift:
                self.particles.remove_drift()

            self.produce()

        time_tot = self.timer.current()
        self.io.time_stamp("Total", self.timer.time_division(time_tot - time0))

        self.directory_sizes()
