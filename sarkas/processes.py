"""
Module handling stages of an MD run: PreProcessing, Simulation, PostProcessing.
"""
import numpy as np
import copy as py_copy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import os

# Sarkas modules
from sarkas.utilities.io import InputOutput
from sarkas.utilities.timing import SarkasTimer
from sarkas.potentials.core import Potential
from sarkas.time_evolution.integrators import Integrator
from sarkas.time_evolution.thermostats import Thermostat
from sarkas.core import Particles, Parameters, Species
from sarkas.utilities.maths import betamp
import sarkas.tools.observables as sk_obs


class Process:
    """Stage of a Molecular Dynamics simulation. This is the Parent class for PreProcess, Simulation, and PostProcess.

    Parameters
    ----------
    input_file : str
        Path to the YAML input file.

    Attributes
    ----------
    potential : sarkas.potential.base.Potential
        Class handling the interaction between particles.

    integrator: sarkas.time_evolution.integrators.Integrator
        Class handling the integrator.

    thermostat: sarkas.time_evolution.thermostats.Thermostat
        Class handling the equilibration thermostat.

    particles: sarkas.core.Particles
        Class handling particles properties.

    parameters: sarkas.core.Parameters
        Class handling simulation's parameters.

    species: list
        List of :meth:`sarkas.core.Species` classes.

    input_file: str
        Path to YAML input file.

    timer: sarkas.utilities.timing.SarkasTimer
        Class handling the timing of processes.

    io: sarkas.utilities.io.InputOutput
        Class handling the IO in Sarkas.

    """

    def __init__(self, input_file: str = None):
        self.potential = Potential()
        self.integrator = Integrator()
        self.thermostat = Thermostat()
        self.parameters = Parameters()
        self.particles = Particles()
        self.species = []
        self.observables_list = []
        self.input_file = input_file if input_file else None
        self.timer = SarkasTimer()
        self.io = InputOutput(process=self.__name__)

    def common_parser(self, filename: str = None) -> None:
        """
        Parse simulation parameters from YAML file.

        Parameters
        ----------
        filename: str
            Input YAML file

        """
        if filename:
            self.input_file = filename

        dics = self.io.from_yaml(self.input_file)

        for lkey in dics:
            if lkey == "Particles":
                for species in dics["Particles"]:
                    spec = Species(species["Species"])
                    self.species.append(spec)

            if lkey == "Potential":
                self.potential.from_dict(dics[lkey])

            if lkey == "Thermostat":
                self.thermostat.from_dict(dics[lkey])

            if lkey == "Integrator":
                self.integrator.from_dict(dics[lkey])

            if lkey == "Parameters":
                self.parameters.from_dict(dics[lkey])

        if self.__name__ != "simulation":
            self.observables_list = []

            for observable in dics["Observables"]:
                for key, sub_dict in observable.items():
                    if key == "RadialDistributionFunction":
                        self.observables_list.append("rdf")
                        self.rdf = sk_obs.RadialDistributionFunction()
                        if sub_dict:
                            self.rdf.from_dict(sub_dict)
                    elif key == "Thermodynamics":
                        self.therm = sk_obs.Thermodynamics()
                        self.therm.from_dict(sub_dict)
                        self.observables_list.append("therm")
                    elif key == "DynamicStructureFactor":
                        self.observables_list.append("dsf")
                        self.dsf = sk_obs.DynamicStructureFactor()
                        if sub_dict:
                            self.dsf.from_dict(sub_dict)
                    elif key == "CurrentCorrelationFunction":
                        self.observables_list.append("ccf")
                        self.ccf = sk_obs.CurrentCorrelationFunction()
                        if sub_dict:
                            self.ccf.from_dict(sub_dict)
                    elif key == "StaticStructureFactor":
                        self.observables_list.append("ssf")
                        self.ssf = sk_obs.StaticStructureFactor()
                        if sub_dict:
                            self.ssf.from_dict(sub_dict)
                    elif key == "VelocityAutoCorrelationFunction":
                        self.observables_list.append("vacf")
                        self.vacf = sk_obs.VelocityAutoCorrelationFunction()
                        if sub_dict:
                            self.vacf.from_dict(sub_dict)
                    elif key == "VelocityDistribution":
                        self.observables_list.append("vd")
                        self.vm = sk_obs.VelocityDistribution()
                        if sub_dict:
                            self.vm.from_dict(sub_dict)
                    elif key == "ElectricCurrent":
                        self.observables_list.append("ec")
                        self.ec = sk_obs.ElectricCurrent()
                        if sub_dict:
                            self.ec.from_dict(sub_dict)
                    elif key == "DiffusionFlux":
                        self.observables_list.append("diff_flux")
                        self.diff_flux = sk_obs.DiffusionFlux()
                        if sub_dict:
                            self.diff_flux.from_dict(sub_dict)
                    elif key == "PressureTensor":
                        self.observables_list.append("p_tensor")
                        self.p_tensor = sk_obs.PressureTensor()
                        if sub_dict:
                            self.p_tensor.from_dict(sub_dict)

            if "TransportCoefficients" in dics.keys():
                self.transport_dict = dics["TransportCoefficients"].copy()

    def initialization(self):
        """Initialize all classes."""

        # initialize the directories and filenames
        self.io.setup()

        # Copy relevant subsclasses attributes into parameters class. This is needed for post-processing.

        # Update parameters' dictionary with filenames and directories
        self.parameters.from_dict(self.io.__dict__)

        self.parameters.potential_type = self.potential.type.lower()

        self.parameters.setup(self.species)

        # save some general info
        self.parameters.integrator = self.integrator.type
        self.parameters.thermostat = self.thermostat.type

        # Copy some integrator parameters if not already defined
        if not hasattr(self.parameters, "dt"):
            self.parameters.dt = self.integrator.dt
        if not hasattr(self.parameters, "equilibration_steps"):
            self.parameters.equilibration_steps = self.integrator.equilibration_steps
        if not hasattr(self.parameters, "eq_dump_step"):
            self.parameters.eq_dump_step = self.integrator.eq_dump_step
        if not hasattr(self.parameters, "production_steps"):
            self.parameters.production_steps = self.integrator.production_steps
        if not hasattr(self.parameters, "prod_dump_step"):
            self.parameters.prod_dump_step = self.integrator.prod_dump_step

        # Check for magnetization phase
        if self.integrator.electrostatic_equilibration:
            self.parameters.electrostatic_equilibration = True
            if not hasattr(self.parameters, "mag_dump_step"):
                self.parameters.mag_dump_step = self.integrator.mag_dump_step
            if not hasattr(self.parameters, "magnetization_steps"):
                self.parameters.magnetization_steps = self.integrator.magnetization_steps

        t0 = self.timer.current()
        self.potential.setup(self.parameters)
        time_pot = self.timer.current()
        self.parameters.cutoff_radius = self.potential.rc

        self.thermostat.setup(self.parameters)
        self.integrator.setup(self.parameters, self.thermostat, self.potential)
        self.particles.setup(self.parameters, self.species)
        time_ptcls = self.timer.current()

        # For restart and backups.
        self.io.setup_checkpoint(self.parameters, self.species)
        self.io.save_pickle(self)

        # Print Process summary to file and screen
        self.io.simulation_summary(self)
        time_end = self.timer.current()

        # Print timing
        self.io.time_stamp("Potential Initialization", self.timer.time_division(time_end - t0))
        self.io.time_stamp("Particles Initialization", self.timer.time_division(time_ptcls - time_pot))
        self.io.time_stamp("Total Simulation Initialization", self.timer.time_division(time_end - t0))

    def setup(self, read_yaml=False, other_inputs=None):
        """Setup simulations' parameters and io subclasses.

        Parameters
        ----------
        read_yaml: bool
            Flag for reading YAML input file. Default = False.

        other_inputs: dict (optional)
            Dictionary with additional simulations options.

        """
        if read_yaml:
            self.common_parser()

        if other_inputs:
            if not isinstance(other_inputs, dict):
                raise TypeError("Wrong input type. " "other_inputs should be a nested dictionary")

            for class_name, class_attr in other_inputs.items():
                if class_name not in ["Particles", "Observables"]:
                    self.__dict__[class_name.lower()].__dict__.update(class_attr)
                elif class_name == "Particles":
                    # Remember Particles should be a list of dict
                    # example:
                    # args = {"Particles" : [ { "Species" : { "name": "O" } } ] }

                    # Check if you already have a non-empty list of species
                    if isinstance(self.species, list):
                        # If so do you want to replace or update?
                        # Update species attributes
                        for sp, species in enumerate(other_inputs["Particles"]):
                            spec = Species(species["Species"])
                            if hasattr(spec, "replace"):
                                self.species[sp].__dict__.update(spec.__dict__)
                            else:
                                self.species.append(spec)
                    else:
                        # Append new species
                        for sp, species in enumerate(other_inputs["Particles"]):
                            spec = Species(species["Species"])
                            self.species.append(spec)

                if class_name == "Observables":

                    for observable in class_attr:
                        for key, sub_dict in observable.items():
                            if key == "RadialDistributionFunction":
                                self.rdf = sk_obs.RadialDistributionFunction()
                                self.rdf.from_dict(sub_dict)
                            if key == "HermiteCoefficients":
                                self.hc = sk_obs.HermiteCoefficients()
                                self.hc.from_dict(sub_dict)
                            if key == "Thermodynamics":
                                self.therm = sk_obs.Thermodynamics()
                                self.therm.from_dict(sub_dict)
                            if key == "DynamicStructureFactor":
                                self.dsf = sk_obs.DynamicStructureFactor()
                                if sub_dict:
                                    self.dsf.from_dict(sub_dict)
                            if key == "CurrentCorrelationFunction":
                                self.ccf = sk_obs.CurrentCorrelationFunction()
                                if sub_dict:
                                    self.ccf.from_dict(sub_dict)
                            if key == "StaticStructureFactor":
                                self.ssf = sk_obs.StaticStructureFactor()
                                if sub_dict:
                                    self.ssf.from_dict(sub_dict)
                            if key == "VelocityAutoCorrelationFunction":
                                self.vacf = sk_obs.VelocityAutoCorrelationFunction()
                                if sub_dict:
                                    self.vacf.from_dict(sub_dict)
                            if key == "VelocityMoments":
                                self.vm = sk_obs.VelocityMoments()
                                if sub_dict:
                                    self.vm.from_dict(sub_dict)
                            if key == "ElectricCurrent":
                                self.ec = sk_obs.ElectricCurrent()
                                if sub_dict:
                                    self.ec.from_dict(sub_dict)

        if self.__name__ == "postprocessing":

            # Create the file paths without creating directories and redefining io attributes
            self.io.create_file_paths()

            # Read previouly stored files
            self.io.read_pickle(self)

            # Print parameters to log file
            if not os.path.exists(self.io.log_file):
                self.io.simulation_summary(self)

            # Initialize the observable classes
            # for obs in self.observables_list:
            #     if obs in self.__dict__.keys():
            #         self.__dict__[obs].setup(self.parameters)

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

    def __init__(self, input_file: str = None):
        self.__name__ = "postprocessing"
        super().__init__(input_file)

    def setup_from_simulation(self, simulation):
        """
        Setup postprocess' subclasses by (shallow) copying them from simulation object.

        Parameters
        ----------
        simulation: sarkas.core.processes.Simulation
            Simulation object

        """
        self.parameters = py_copy.copy(simulation.parameters)
        self.integrator = py_copy.copy(simulation.integrator)
        self.potential = py_copy.copy(simulation.potential)
        self.species = py_copy.copy(simulation.species)
        self.thermostat = py_copy.copy(simulation.thermostat)
        self.io = py_copy.copy(simulation.io)

    def run(self):
        """Calculate all the observables from the YAML input file."""

        if len(self.observables_list) == 0:
            # Make Temperature and Energy plots
            self.therm = sk_obs.Thermodynamics()
            self.therm.setup(self.parameters)
            if self.parameters.equilibration_steps > 0:
                self.therm.temp_energy_plot(self, phase="equilibration")
            self.therm.temp_energy_plot(self, phase="production")
            # Calculate the RDF.
            self.rdf = sk_obs.RadialDistributionFunction()
            self.rdf.setup(self.parameters)
            self.rdf.parse()
        else:
            for obs in self.observables_list:
                # Check that the observable is actually there
                if obs in self.__dict__.keys():
                    self.__dict__[obs].setup(self.parameters)
                    if obs == "therm":
                        self.therm.temp_energy_plot(self)
                    else:
                        self.io.postprocess_info(self, write_to_file=True, observable=obs)
                        self.__dict__[obs].compute()

                # Calculate transport coefficients
                if hasattr(self, "transport_dict"):
                    from sarkas.tools.transport import TransportCoefficients

                    tc = TransportCoefficients(self.parameters)

                    for coeff in self.transport_dict:

                        for key, coeff_kwargs in coeff.items():

                            if key.lower() == "diffusion":
                                # Calculate if not already
                                if not self.vacf:
                                    self.vacf = sk_obs.VelocityAutoCorrelationFunction()
                                    self.vacf.setup(self.parameters)
                                    # Use parse in case you calculated it already
                                    self.vacf.parse()

                                tc.diffusion(observable=self.vacf)

                            elif key.lower() == "interdiffusion":
                                if not self.diff_flux:
                                    self.diff_flux = sk_obs.DiffusionFlux()
                                    self.diff_flux.setup(self.parameters)
                                    self.diff_flux.parse()

                                tc.interdiffusion(self.diff_flux)

                            elif key.lower() == "viscosity":
                                if not self.p_tensor:
                                    self.p_tensor = sk_obs.PressureTensor()
                                    self.p_tensor.setup(self.parameters)
                                    self.p_tensor.parse()
                                tc.viscosity(self.p_tensor)

                            elif key.lower() == "electricalconductivity":
                                if not self.ec:
                                    self.ec = sk_obs.ElectricCurrent()
                                    self.ec.setup(self.parameters)
                                    self.ec.parse()

                                tc.electrical_conductivity(self.ec)


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
        self.loops = 10
        self.estimate = False
        self.pm_meshes = np.logspace(3, 7, 12, base=2, dtype=int)
        # np.array([16, 24, 32, 48, 56, 64, 72, 88, 96, 112, 128], dtype=int)
        self.pp_cells = np.arange(3, 16, dtype=int)
        self.kappa = None
        super().__init__(input_file)

    def green_function_timer(self):
        """Time Potential setup."""

        self.timer.start()
        self.potential.pppm_setup(self.parameters)

        return self.timer.stop()

    def run(
        self,
        loops: int = None,
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
            Number of loops over which to average the acceleration calculation.
            Note that the number of timestep over which to averages is three times this value.
            Example: loops = 5, acceleration is averaged over 5 loops, while full time step over 15 loops.

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
        if pppm_estimate:
            self.pppm_plots_dir = os.path.join(self.io.preprocessing_dir, "PPPM_Plots")
            if not os.path.exists(self.pppm_plots_dir):
                os.mkdir(self.pppm_plots_dir)

        # Set the screening parameter
        self.kappa = self.potential.matrix[1, 0, 0] if self.potential.type == "yukawa" else 0.0

        if loops:
            self.loops = loops + 1

        if timing:
            self.io.preprocess_timing("header", [0, 0, 0, 0, 0, 0], 0)
            if self.potential.pppm_on:
                green_time = self.timer.time_division(self.green_function_timer())
                self.io.preprocess_timing("GF", green_time, 0)

            self.time_acceleration()
            self.time_integrator_loop()

            # Estimate size of dump folder
            # Grab one file from the dump directory and get the size of it.
            eq_dump_size = os.stat(os.path.join(self.io.eq_dump_dir, os.listdir(self.io.eq_dump_dir)[0])).st_size
            eq_dump_fldr_size = eq_dump_size * (self.integrator.equilibration_steps / self.integrator.eq_dump_step)
            # Grab one file from the dump directory and get the size of it.
            prod_dump_size = os.stat(os.path.join(self.io.eq_dump_dir, os.listdir(self.io.eq_dump_dir)[0])).st_size
            prod_dump_fldr_size = prod_dump_size * (self.integrator.production_steps / self.integrator.prod_dump_step)
            # Prepare arguments to pass for print out
            sizes = np.array([[eq_dump_size, eq_dump_fldr_size], [prod_dump_size, prod_dump_fldr_size]])
            # Check for electrostatic equilibration
            if self.integrator.electrostatic_equilibration:
                dump = self.integrator.mag_dump_step
                mag_dump_size = os.stat(os.path.join(self.io.mag_dump_dir, "checkpoint_" + str(dump) + ".npz")).st_size
                mag_dump_fldr_size = mag_dump_size * (self.integrator.magnetization_steps / self.integrator.mag_dump_step)
                sizes = np.array(
                    [
                        [eq_dump_size, eq_dump_fldr_size],
                        [prod_dump_size, prod_dump_fldr_size],
                        [mag_dump_size, mag_dump_fldr_size],
                    ]
                )

            self.io.preprocess_sizing(sizes)

            if remove:
                # Delete the energy files created during the estimation runs
                os.remove(self.io.eq_energy_filename)
                os.remove(self.io.prod_energy_filename)

                # Delete dumps created during the estimation runs
                for npz in os.listdir(self.io.eq_dump_dir):
                    os.remove(os.path.join(self.io.eq_dump_dir, npz))

                for npz in os.listdir(self.io.prod_dump_dir):
                    os.remove(os.path.join(self.io.prod_dump_dir, npz))

                if self.integrator.electrostatic_equilibration:
                    os.remove(self.io.mag_energy_filename)
                    # Remove dumps
                    for npz in os.listdir(self.io.mag_dump_dir):
                        os.remove(os.path.join(self.io.mag_dump_dir, npz))

        if pppm_estimate:
            if timing_study:
                self.input_rc = self.potential.rc
                self.input_mesh = np.copy(self.potential.pppm_mesh)
                self.input_alpha = self.potential.pppm_alpha_ewald

                self.timing_study = timing_study
                self.make_timing_plots()

                # Reset the original values.
                self.potential.rc = self.input_rc
                self.potential.pppm_mesh = np.copy(self.input_mesh)
                self.potential.pppm_alpha_ewald = self.input_alpha
                self.potential.setup(self.parameters)

            self.pppm_approximation()
            print("\nFigures can be found in {}".format(self.pppm_plots_dir))

        if postprocessing:
            # POST- PROCESSING
            self.io.postprocess_info(self, write_to_file=True, observable="header")

            if hasattr(self, "rdf"):
                self.rdf.setup(self.parameters)
                self.io.postprocess_info(self, write_to_file=True, observable="rdf")

            if hasattr(self, "ssf"):
                self.ssf.setup(self.parameters)
                self.io.postprocess_info(self, write_to_file=True, observable="ssf")

            if hasattr(self, "dsf"):
                self.dsf.setup(self.parameters)
                self.io.postprocess_info(self, write_to_file=True, observable="dsf")

            if hasattr(self, "ccf"):
                self.ccf.setup(self.parameters)
                self.io.postprocess_info(self, write_to_file=True, observable="ccf")

            if hasattr(self, "vm"):
                self.ccf.setup(self.parameters)
                self.io.postprocess_info(self, write_to_file=True, observable="vm")

    def make_timing_plots(self):
        """Estimate the best number of mesh points and cutoff radius."""

        from scipy.optimize import curve_fit

        print("\n\n{:=^70} \n".format(" Timing Study "))

        max_cells = int(0.5 * self.parameters.box_lengths.min() / self.parameters.a_ws)
        if max_cells != self.pp_cells[-1]:
            self.pp_cells = np.arange(3, max_cells, dtype=int)

        pm_times = np.zeros(len(self.pm_meshes))
        pm_errs = np.zeros(len(self.pm_meshes))

        pp_times = np.zeros((len(self.pm_meshes), len(self.pp_cells)))
        pp_errs = np.zeros((len(self.pm_meshes), len(self.pp_cells)))

        pm_xlabels = []
        pp_xlabels = []

        self.force_error_map = np.zeros((len(self.pm_meshes), len(self.pp_cells)))

        # Average the PM time
        for i, m in enumerate(self.pm_meshes):

            self.potential.pppm_mesh = m * np.ones(3, dtype=int)
            self.potential.pppm_alpha_ewald = 0.3 * m / self.parameters.box_lengths.min()
            green_time = self.green_function_timer()
            pm_errs[i] = self.parameters.pppm_pm_err
            print("\n\nMesh = {} x {} x {} : ".format(*self.potential.pppm_mesh))
            print(
                "alpha = {:.4f} / a_ws = {:.4e} ".format(
                    self.potential.pppm_alpha_ewald * self.parameters.a_ws, self.potential.pppm_alpha_ewald
                )
            )
            print("PM Err = {:.6e}".format(self.parameters.pppm_pm_err))

            self.io.preprocess_timing("GF", self.timer.time_division(green_time), 0)
            pm_xlabels.append("{}x{}x{}".format(*self.potential.pppm_mesh))

            # Calculate the PM acceleration timing 3x and average
            for it in range(3):
                self.timer.start()
                self.potential.update_pm(self.particles)
                pm_times[i] += self.timer.stop() / 3.0

            # For each number of PP cells, calculate the PM acceleration timing 3x and average
            for j, c in enumerate(self.pp_cells):
                self.potential.rc = self.parameters.box_lengths.min() / c
                kappa_over_alpha = -0.25 * (self.kappa / self.potential.pppm_alpha_ewald) ** 2
                alpha_times_rcut = -((self.potential.pppm_alpha_ewald * self.potential.rc) ** 2)
                # Update the Force error
                self.potential.pppm_pp_err = (
                    2.0 * np.exp(kappa_over_alpha + alpha_times_rcut) / np.sqrt(self.potential.rc)
                )
                self.potential.pppm_pp_err *= (
                    np.sqrt(self.parameters.total_num_ptcls)
                    * self.parameters.a_ws ** 2
                    / np.sqrt(self.parameters.box_volume)
                )

                pp_errs[i, j] = self.potential.pppm_pp_err
                self.force_error_map[i, j] = np.sqrt(self.potential.pppm_pp_err ** 2 + self.parameters.pppm_pm_err ** 2)

                if j == 0:
                    pp_xlabels.append("{:.2f}".format(self.potential.rc / self.parameters.a_ws))

                for it in range(3):
                    self.timer.start()
                    self.potential.update_linked_list(self.particles)
                    pp_times[i, j] += self.timer.stop() / 3.0

        # Get the time in seconds
        pp_times *= 1e-9
        pm_times *= 1e-9
        # Fit the PM times
        pm_popt, _ = curve_fit(lambda x, a, b: a + 5 * b * x ** 3 * np.log2(x ** 3), self.pm_meshes, pm_times)
        fit_str = r"Fit = $a_2 + 5 a_3 M^3 \log_2(M^3)$  [s]" + "\n" + r"$a_2 = ${:.4e}, $a_3 = ${:.4e} ".format(*pm_popt)
        print("\nPM Time " + fit_str)

        # Fit the PP Times
        pp_popt, _ = curve_fit(
            lambda x, a, b: a + b / x ** 3,
            self.pp_cells,
            np.mean(pp_times, axis=0),
            p0=[np.mean(pp_times, axis=0)[0], self.parameters.total_num_ptcls],
            bounds=(0, [np.mean(pp_times, axis=0)[0], 1e9]),
        )
        fit_pp_str = r"Fit = $a_0 + a_1 / N_c^3$  [s]" + "\n" + "$a_0 = ${:.4e},  $a_1 = ${:.4e}".format(*pp_popt)
        print("\nPP Time " + fit_pp_str)

        # Start the plot
        fig, (ax_pp, ax_pm) = plt.subplots(1, 2, sharey=True, figsize=(12, 7))
        ax_pm.plot(self.pm_meshes, pm_times, "o", label="Measured")
        ax_pm.plot(
            self.pm_meshes,
            pm_popt[0] + 5 * pm_popt[1] * self.pm_meshes ** 3 * np.log2(self.pm_meshes ** 3),
            ls="--",
            label="Fit",
        )
        ax_pm.set(title="PM calculation time and estimate", yscale="log", xlabel="Mesh size")
        ax_pm.set_xscale("log", base=2)
        ax_pm.legend(ncol=2)
        ax_pm.annotate(
            text=fit_str,
            xy=(self.pm_meshes[-1], pm_times[-1]),
            xytext=(self.pm_meshes[0], pm_times[-1]),
            bbox=dict(boxstyle="round4", fc="white", ec="k", lw=2),
        )

        # Scatter Plot the PP Times
        self.tot_time_map = np.zeros(pp_times.shape)
        for j, mesh_points in enumerate(self.pm_meshes):
            self.tot_time_map[j, :] = pm_times[j] + pp_times[j, :]
            ax_pp.plot(self.pp_cells, pp_times[j], "o", label=r"@ Mesh {}$^3$".format(mesh_points))

        # Plot the Fit PP times
        ax_pp.plot(self.pp_cells, pp_popt[0] + pp_popt[1] / self.pp_cells ** 3, ls="--", label="Fit")
        ax_pp.legend(ncol=2)
        ax_pp.annotate(
            text=fit_pp_str,
            xy=(self.pp_cells[0], pp_times[0, 0]),
            xytext=(self.pp_cells[0], pp_times[-1, -1]),
            bbox=dict(boxstyle="round4", fc="white", ec="k", lw=2),
        )
        ax_pp.set(
            title="PP calculation time and estimate", yscale="log", ylabel="CPU Times [s]", xlabel=r"$N_c $ = Cells"
        )
        fig.tight_layout()
        fig.savefig(os.path.join(self.pppm_plots_dir, "Times_" + self.io.job_id + ".png"))

        self.make_force_v_timing_plot()
        # self.lagrangian = np.empty((len(self.pm_meshes), len(self.pp_cells)))
        # self.tot_times = np.empty((len(self.pm_meshes), len(self.pp_cells)))
        # self.pp_times = np.copy(pp_times)
        # self.pm_times = np.copy(pm_times)
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
        #                    self.timer.time_division(self.predicted_times * self.integrator.equilibration_steps))
        # self.io.time_stamp('Production',
        #                    self.timer.time_division(self.predicted_times * self.integrator.production_steps))
        # self.io.time_stamp('Total Run',
        #                    self.timer.time_division(self.predicted_times * (self.integrator.equilibration_steps
        #                                                                     + self.integrator.production_steps)))

    def make_lagrangian_plot(self):

        c_mesh, m_mesh = np.meshgrid(self.pp_cells, self.pm_meshes)
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
        fig.savefig(os.path.join(self.io.preprocessing_dir, "2D_Lagrangian.png"))

    def make_force_v_timing_plot(self):

        # Plot the results
        fig_path = self.pppm_plots_dir
        c_mesh, m_mesh = np.meshgrid(self.pp_cells, self.pm_meshes)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
        if self.force_error_map.min() == 0.0:
            minv = 1e-120
        else:
            minv = self.force_error_map.min()

        maxt = self.force_error_map.max()
        nlvl = 12
        lvls = np.logspace(np.log10(minv), np.log10(maxt), nlvl)

        luxmap = mpl.cm.get_cmap("viridis", nlvl)
        luxnorm = mpl.colors.LogNorm(vmin=minv, vmax=maxt)
        CS = ax1.contourf(m_mesh, c_mesh, self.force_error_map, levels=lvls, cmap=luxmap, norm=luxnorm)
        clb = fig.colorbar(mpl.cm.ScalarMappable(norm=luxnorm, cmap=luxmap), ax=ax1)
        clb.set_label(r"Force Error  [$Q^2/ a_{\rm ws}^2$]", rotation=270, va="bottom")
        CS2 = ax1.contour(CS, colors="w")
        ax1.clabel(CS2, fmt="%1.0e", colors="w")
        input_Nc = int(self.parameters.box_lengths[0] / self.input_rc)
        ax1.scatter(self.input_mesh[0], input_Nc, s=200, c="k")
        # ax1.scatter(self.input_mesh[1], input_Nc, s=200, c='k')
        # ax1.scatter(self.input_mesh[2], input_Nc, s=200, c='k')
        ax1.set_xlabel("Mesh size")
        ax1.set_ylabel(r"N_c = Cells")
        ax1.set_title("Force Error Map")

        # Timing Plot
        maxt = self.tot_time_map.max()
        mint = self.tot_time_map.min()
        # nlvl = 13
        lvls = np.logspace(np.log10(mint), np.log10(maxt), nlvl)
        luxmap = mpl.cm.get_cmap("viridis", nlvl)
        luxnorm = mpl.colors.LogNorm(vmin=minv, vmax=maxt)

        CS = ax2.contourf(m_mesh, c_mesh, self.tot_time_map, levels=lvls, cmap=luxmap)
        CS2 = ax2.contour(CS, colors="w", levels=lvls)
        ax2.clabel(CS2, fmt="%.2e", colors="w")
        # fig.colorbar(, ax = ax2)
        clb = fig.colorbar(mpl.cm.ScalarMappable(norm=luxnorm, cmap=luxmap), ax=ax2)
        clb.set_label("CPU Time [s]", rotation=270, va="bottom")
        ax2.scatter(self.input_mesh[0], input_Nc, s=200, c="k")
        ax2.set_xlabel("Mesh size")
        ax2.set_title("Timing Map")
        fig.savefig(os.path.join(fig_path, "ForceErrorMap_v_Timing_" + self.io.job_id + ".png"))

    def time_acceleration(self):

        self.pp_acc_time = np.zeros(self.loops)
        for i in range(self.loops):
            self.timer.start()
            self.potential.update_linked_list(self.particles)
            self.pp_acc_time[i] = self.timer.stop()

        # Calculate the mean excluding the first value because that time include numba compilation time
        pp_mean_time = self.timer.time_division(np.mean(self.pp_acc_time[1:]))

        self.io.preprocess_timing("PP", pp_mean_time, self.loops)

        # PM acceleration
        if self.potential.pppm_on:
            self.pm_acc_time = np.zeros(self.loops)
            for i in range(self.loops):
                self.timer.start()
                self.potential.update_pm(self.particles)
                self.pm_acc_time[i] = self.timer.stop()
            pm_mean_time = self.timer.time_division(np.mean(self.pm_acc_time[1:]))
            self.io.preprocess_timing("PM", pm_mean_time, self.loops)

    def time_integrator_loop(self):
        """Run several loops of the equilibration and production phase to estimate the total time of the simulation."""
        if self.parameters.electrostatic_equilibration:
            # Save the original number of timesteps
            steps = np.array(
                [
                    self.integrator.equilibration_steps,
                    self.integrator.production_steps,
                    self.integrator.magnetization_steps,
                ]
            )
            self.integrator.magnetization_steps = self.loops
        else:
            # Save the original number of timesteps
            steps = np.array([self.integrator.equilibration_steps, self.integrator.production_steps])

        # Update the equilibration and production timesteps for estimation
        self.integrator.production_steps = self.loops
        self.integrator.equilibration_steps = self.loops

        if self.io.verbose:
            print("\nRunning {} equilibration and production steps to estimate simulation times\n".format(self.loops))

        # Run few equilibration steps to estimate the equilibration time
        self.timer.start()
        self.integrator.equilibrate(0, self.particles, self.io)
        self.eq_mean_time = self.timer.stop() / self.loops
        # Print the average equilibration & production times
        self.io.preprocess_timing("Equilibration", self.timer.time_division(self.eq_mean_time), self.loops)

        if self.integrator.electrostatic_equilibration:
            self.timer.start()
            self.integrator.magnetize(0, self.particles, self.io)
            self.mag_mean_time = self.timer.stop() / self.loops
            # Print the average equilibration & production times
            self.io.preprocess_timing("Magnetization", self.timer.time_division(self.mag_mean_time), self.loops)

        # Run few production steps to estimate the equilibration time
        self.timer.start()
        self.integrator.produce(0, self.particles, self.io)
        self.prod_mean_time = self.timer.stop() / self.loops
        self.io.preprocess_timing("Production", self.timer.time_division(self.prod_mean_time), self.loops)

        # Restore the original number of timesteps and print an estimate of run times
        self.integrator.equilibration_steps = steps[0]
        self.integrator.production_steps = steps[1]

        # Print the estimate for the full run
        eq_prediction = self.eq_mean_time * steps[0]
        self.io.time_stamp("Equilibration", self.timer.time_division(eq_prediction))

        if self.integrator.electrostatic_equilibration:
            self.integrator.magnetization_steps = steps[2]
            mag_prediction = self.mag_mean_time * steps[2]
            self.io.time_stamp("Magnetization", self.timer.time_division(mag_prediction))
            eq_prediction += mag_prediction

        prod_prediction = self.prod_mean_time * steps[1]
        self.io.time_stamp("Production", self.timer.time_division(prod_prediction))

        tot_time = eq_prediction + prod_prediction
        self.io.time_stamp("Total Run", self.timer.time_division(tot_time))

    def pppm_approximation(self):
        """
        Calculate the Force error for a PPPM simulation
        using analytical approximations.
        Plot the force error in the parameter space.
        """

        # Calculate Force error from analytic approximation given in Dharuman et al. J Chem Phys 2017
        total_force_error, pp_force_error, pm_force_error, rcuts, alphas = self.analytical_approx_pppm()

        chosen_alpha = self.potential.pppm_alpha_ewald * self.parameters.a_ws
        chosen_rcut = self.potential.rc / self.parameters.a_ws

        # mesh_dir = os.path.join(self.pppm_plots_dir, 'Mesh_{}'.format(self.potential.pppm_mesh[0]))
        # if not os.path.exists(mesh_dir):
        #     os.mkdir(mesh_dir)
        #
        # cell_num = int(self.parameters.box_lengths.min() / self.potential.rc)
        # cell_dir = os.path.join(mesh_dir, 'Cells_{}'.format(cell_num))
        # if not os.path.exists(cell_dir):
        #     os.mkdir(cell_dir)
        #
        # self.pppm_plots_dir = cell_dir

        # Color Map
        self.make_color_map(rcuts, alphas, chosen_alpha, chosen_rcut, total_force_error)

        # Line Plot
        self.make_line_plot(rcuts, alphas, chosen_alpha, chosen_rcut, total_force_error)

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

        """
        # Plot the results
        fig_path = self.pppm_plots_dir

        fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 7))
        ax[0].plot(
            rcuts, total_force_error[30, :], ls=(0, (5, 10)), label=r"$\alpha a_{ws} = " + "{:2.2f}$".format(alphas[30])
        )
        ax[0].plot(
            rcuts, total_force_error[40, :], ls="dashed", label=r"$\alpha a_{ws} = " + "{:2.2f}$".format(alphas[40])
        )
        ax[0].plot(
            rcuts, total_force_error[50, :], ls="solid", label=r"$\alpha a_{ws} = " + "{:2.2f}$".format(alphas[50])
        )
        ax[0].plot(
            rcuts, total_force_error[60, :], ls="dashdot", label=r"$\alpha a_{ws} = " + "{:2.2f}$".format(alphas[60])
        )
        ax[0].plot(
            rcuts,
            total_force_error[70, :],
            ls=(0, (3, 10, 1, 10)),
            label=r"$\alpha a_{ws} = " + "{:2.2f}$".format(alphas[70]),
        )
        ax[0].set_ylabel(r"$\Delta F^{approx}_{tot}$")
        ax[0].set_xlabel(r"$r_c/a_{ws}$")
        ax[0].set_yscale("log")
        ax[0].axvline(chosen_rcut, ls="--", c="k")
        ax[0].axhline(self.parameters.force_error, ls="--", c="k")
        if rcuts[-1] * self.parameters.a_ws > 0.5 * self.parameters.box_lengths.min():
            ax[0].axvline(0.5 * self.parameters.box_lengths.min() / self.parameters.a_ws, c="r", label=r"$L/2$")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend(loc="best")

        ax[1].plot(
            alphas, total_force_error[:, 30], ls=(0, (5, 10)), label=r"$r_c = {:2.2f}".format(rcuts[30]) + " a_{ws}$"
        )
        ax[1].plot(alphas, total_force_error[:, 40], ls="dashed", label=r"$r_c = {:2.2f}".format(rcuts[40]) + " a_{ws}$")
        ax[1].plot(alphas, total_force_error[:, 50], ls="solid", label=r"$r_c = {:2.2f}".format(rcuts[50]) + " a_{ws}$")
        ax[1].plot(alphas, total_force_error[:, 60], ls="dashdot", label=r"$r_c = {:2.2f}".format(rcuts[60]) + " a_{ws}$")
        ax[1].plot(
            alphas,
            total_force_error[:, 70],
            ls=(0, (3, 10, 1, 10)),
            label=r"$r_c = {:2.2f}".format(rcuts[70]) + " a_{ws}$",
        )
        ax[1].set_xlabel(r"$\alpha \; a_{ws}$")
        ax[1].set_yscale("log")
        ax[1].axhline(self.parameters.force_error, ls="--", c="k")
        ax[1].axvline(chosen_alpha, ls="--", c="k")
        ax[1].grid(True, alpha=0.3)
        ax[1].legend(loc="best")
        fig.suptitle(
            r"Parameters  $N = {}, \quad M = {}, \quad p = {}, \quad \kappa = {:.2f}$".format(
                self.parameters.total_num_ptcls,
                self.potential.pppm_mesh[0],
                self.potential.pppm_cao,
                self.kappa * self.parameters.a_ws,
            )
        )
        fig.savefig(os.path.join(fig_path, "LinePlot_ForceError_" + self.io.job_id + ".png"))

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
        """
        # Plot the results
        fig_path = self.pppm_plots_dir

        r_mesh, a_mesh = np.meshgrid(rcuts, alphas)
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        if total_force_error.min() == 0.0:
            minv = 1e-120
        else:
            minv = total_force_error.min()
        total_force_error[total_force_error == 0.0] = minv
        CS = ax.contourf(a_mesh, r_mesh, total_force_error, norm=LogNorm(vmin=minv, vmax=total_force_error.max()))
        CS2 = ax.contour(CS, colors="w")
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
                self.potential.pppm_cao,
                self.kappa * self.parameters.a_ws,
            )
        )
        clb = fig.colorbar(CS)
        clb.set_label(r"$\Delta F^{approx}_{tot}(r_c,\alpha)$", va="bottom", rotation=270)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_path, "ClrMap_ForceError_" + self.io.job_id + ".png"))

    def analytical_approx_pp(self):
        """Calculate PP force error."""

        r_min = self.potential.rc * 0.5
        r_max = self.potential.rc * 1.5

        rcuts = np.linspace(r_min, r_max, 101) / self.parameters.a_ws

        # Calculate the analytic PP error and the total force error
        pp_force_error = np.sqrt(2.0 * np.pi * self.kappa) * np.exp(-rcuts * self.kappa)
        pp_force_error *= np.sqrt(
            self.parameters.total_num_ptcls * self.parameters.a_ws ** 3 / self.parameters.box_volume
        )

        return pp_force_error, rcuts

    def analytical_approx_pppm(self):
        """Calculate the total force error as given in :cite:`Dharuman2017`."""

        p = self.potential.pppm_cao
        L = self.parameters.box_lengths[0] / self.parameters.a_ws
        h = L / self.potential.pppm_mesh[0]

        a_min = self.potential.pppm_alpha_ewald * 0.5
        a_max = self.potential.pppm_alpha_ewald * 1.5

        r_min = self.potential.rc * 0.5
        r_max = self.potential.rc * 1.5

        alphas = self.parameters.a_ws * np.linspace(a_min, a_max, 101)
        rcuts = np.linspace(r_min, r_max, 101) / self.parameters.a_ws

        pm_force_error = np.zeros(len(alphas))
        pp_force_error = np.zeros((len(alphas), len(rcuts)))
        total_force_error = np.zeros((len(alphas), len(rcuts)))

        # Coefficient from Deserno and Holm J Chem Phys 109 7694 (1998)
        if p == 1:
            Cmp = np.array([2 / 3])
        elif p == 2:
            Cmp = np.array([2 / 45, 8 / 189])
        elif p == 3:
            Cmp = np.array([4 / 495, 2 / 225, 8 / 1485])
        elif p == 4:
            Cmp = np.array([2 / 4725, 16 / 10395, 5528 / 3869775, 32 / 42525])
        elif p == 5:
            Cmp = np.array([4 / 93555, 2764 / 11609325, 8 / 25515, 7234 / 32531625, 350936 / 3206852775])
        elif p == 6:
            Cmp = np.array(
                [
                    2764 / 638512875,
                    16 / 467775,
                    7234 / 119282625,
                    1403744 / 25196700375,
                    1396888 / 40521009375,
                    2485856 / 152506344375,
                ]
            )
        elif p == 7:
            Cmp = np.array(
                [
                    8 / 18243225,
                    7234 / 1550674125,
                    701872 / 65511420975,
                    2793776 / 225759909375,
                    1242928 / 132172165125,
                    1890912728 / 352985880121875,
                    21053792 / 8533724574375,
                ]
            )

        kappa = self.kappa * self.parameters.a_ws

        for ia, alpha in enumerate(alphas):
            somma = 0.0
            for m in np.arange(p):
                expp = 2 * (m + p)
                somma += Cmp[m] * (2.0 / (1 + expp)) * betamp(m, p, alpha, kappa) * (h / 2.0) ** expp
            # eq.(36) in Dharuman J Chem Phys 146 024112 (2017)
            pm_force_error[ia] = np.sqrt(3.0 * somma) / (2.0 * np.pi)
        # eq.(35)
        pm_force_error *= np.sqrt(
            self.parameters.total_num_ptcls * self.parameters.a_ws ** 3 / self.parameters.box_volume
        )
        # Calculate the analytic PP error and the total force error
        if self.potential.type == "qsp":
            for (ir, rc) in enumerate(rcuts):
                pp_force_error[:, ir] = np.sqrt(2.0 * np.pi * kappa) * np.exp(-rc * kappa)
                pp_force_error[:, ir] *= np.sqrt(
                    self.parameters.total_num_ptcls * self.parameters.a_ws ** 3 / self.parameters.box_volume
                )
                for (ia, alfa) in enumerate(alphas):
                    # eq.(42) from Dharuman J Chem Phys 146 024112 (2017)
                    total_force_error[ia, ir] = np.sqrt(pm_force_error[ia] ** 2 + pp_force_error[ia, ir] ** 2)
        else:
            for (ir, rc) in enumerate(rcuts):
                for (ia, alfa) in enumerate(alphas):
                    # eq.(30) from Dharuman J Chem Phys 146 024112 (2017)
                    pp_force_error[ia, ir] = (
                        2.0 * np.exp(-((0.5 * kappa / alfa) ** 2) - alfa ** 2 * rc ** 2) / np.sqrt(rc)
                    )
                    pp_force_error[ia, ir] *= np.sqrt(
                        self.parameters.total_num_ptcls * self.parameters.a_ws ** 3 / self.parameters.box_volume
                    )
                    # eq.(42) from Dharuman J Chem Phys 146 024112 (2017)
                    total_force_error[ia, ir] = np.sqrt(pm_force_error[ia] ** 2 + pp_force_error[ia, ir] ** 2)

        return total_force_error, pp_force_error, pm_force_error, rcuts, alphas

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
        fig.savefig(os.path.join(fig_path, "Timing_Fit.png"))


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

    def equilibrate(self) -> None:
        """
        Run the time integrator with the thermostat to evolve the system to its thermodynamics equilibrium state.
        """
        if self.parameters.verbose:
            print("\n\n{:-^70} \n".format(" Equilibration "))
        # Check if this is restart
        if self.parameters.load_method in ["equilibration_restart", "eq_restart"]:
            it_start = self.parameters.restart_step
        else:
            it_start = 0
            self.io.dump("equilibration", self.particles, 0)

        # Start timer, equilibrate, and print run time.
        self.timer.start()
        self.integrator.equilibrate(it_start, self.particles, self.io)
        time_eq = self.timer.stop()
        self.io.time_stamp("Equilibration", self.timer.time_division(time_eq))

        # Check for magnetization phase
        if self.integrator.electrostatic_equilibration:
            if self.parameters.verbose:
                print("\n\n{:-^70} \n".format(" Magnetization "))

            if self.parameters.load_method in ["magnetization_restart", "mag_restart"]:
                it_start = self.parameters.restart_step
            else:
                it_start = 0
                self.io.dump("magnetization", self.particles, it_start)

            # Start timer, magnetize, and print run time.
            self.timer.start()
            self.integrator.magnetize(it_start, self.particles, self.io)
            time_eq = self.timer.stop()
            self.io.time_stamp("Magnetization", self.timer.time_division(time_eq))

    def evolve(self) -> None:
        """
        Run the time integrator to evolve the system for the duration of the production phase.
        """

        # Check for simulation restart.
        if self.parameters.load_method in ["prod_restart", "production_restart"]:
            it_start = self.parameters.restart_step
        else:
            it_start = 0
            # Restart the pbc counter.
            self.particles.pbc_cntr.fill(0)
            self.io.dump("production", self.particles, 0)

        if self.parameters.verbose:
            print("\n\n{:-^70} \n".format(" Production "))

        # Update measurement flag for rdf.
        self.potential.measure = True

        # Start timer, produce data, and print run time.
        self.timer.start()
        self.integrator.produce(it_start, self.particles, self.io)
        time_end = self.timer.stop()
        self.io.time_stamp("Production", self.timer.time_division(time_end))

    def run(self) -> None:
        """Run the simulation."""
        time0 = self.timer.current()

        if not self.parameters.load_method in ["prod_restart", "production_restart"]:
            self.equilibrate()

        self.evolve()
        time_tot = self.timer.current()
        self.io.time_stamp("Total", self.timer.time_division(time_tot - time0))
