import os
import sys
import re
import yaml
import csv
import pickle
import numpy as np
from pyfiglet import print_figlet, Figlet
from IPython import get_ipython

if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    # If you are using Jupyter Notebook
    from tqdm import tqdm_notebook as tqdm
else:
    # If you are using IPython or Python kernel
    from tqdm import tqdm

FONTS = ["speed", "starwars", "graffiti", "chunky", "epic", "larry3d", "ogre"]

# Light Colors.
LIGHT_COLORS = [
    "255;255;255",
    "13;177;75",
    "153;162;162",
    "240;133;33",
    "144;154;183",
    "209;222;63",
    "232;217;181",
    "200;154;88",
    "148;174;74",
    "203;90;40",
]

# Dark Colors.
DARK_COLORS = ["24;69;49", "0;129;131", "83;80;84", "110;0;95"]


class InputOutput:
    def __init__(self, process: str = None):
        """Set default directory names."""
        self.process = process if process else "preprocessing"
        self.input_file = None
        self.equilibration_dir = "Equilibration"
        self.production_dir = "Production"
        self.magnetization_dir = "Magnetization"
        self.simulations_dir = "Simulations"
        self.processes_dir = None
        self.simulation_dir = "Simulation"
        self.preprocessing_dir = "PreProcessing"
        self.postprocessing_dir = "PostProcessing"
        self.prod_dump_dir = "dumps"
        self.eq_dump_dir = "dumps"
        self.mag_dump_dir = "dumps"
        self.job_dir = None
        self.job_id = None
        self.log_file = None
        self.preprocess_file = None
        self.preprocessing = False
        self.magnetized = False
        self.electrostatic_equilibration = False
        self.verbose = False
        self.xyz_dir = None
        self.xyz_filename = None

    def __repr__(self):
        sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
        disp = "InputOuput( \n"
        for key, value in sortedDict.items():
            disp += "\t{} : {}\n".format(key, value)
        disp += ")"
        return disp

    def from_dict(self, input_dict: dict):
        """
        Update attributes from input dictionary.

        Parameters
        ----------
        input_dict: dict
            Dictionary to be copied.

        """
        self.__dict__.update(input_dict)

    def setup(self):
        """Create file paths and directories for the simulation."""
        self.create_file_paths()
        self.make_directories()
        self.file_header()

    def from_yaml(self, filename):
        """
        Parse inputs from YAML file.

        Parameters
        ----------
        filename: str
            Input YAML file.

        Returns
        -------
        dics : dict
            Content of YAML file parsed in a nested dictionary
        """

        self.input_file = filename
        with open(filename, "r") as stream:
            dics = yaml.load(stream, Loader=yaml.FullLoader)
            self.__dict__.update(dics["IO"])

        if "Parameters" in dics.keys():
            keyed = "Parameters"
            for key, value in dics[keyed].items():

                if key == "verbose":
                    self.verbose = value

                if key == "magnetized":
                    self.magnetized = value

                if key == "load_method":
                    self.load_method = value
                    if value[-7:] == "restart":
                        self.restart = True
                    else:
                        self.restart = False

                if key == "preprocessing":
                    self.preprocessing = value

        if "Integrator" in dics.keys():
            keyed = "Integrator"
            for key, value in dics[keyed].items():

                if key == "electrostatic_equilibration":
                    self.electrostatic_equilibration = value

        # rdf_nbins can be defined in either Parameters or Postprocessing. However, Postprocessing will always
        # supersede Parameters choice.
        if "Observables" in dics.keys():
            for i in dics["Observables"]:
                if "RadialDistributionFunction" in i.keys():
                    dics["Parameters"]["rdf_nbins"] = i["RadialDistributionFunction"]["no_bins"]

        return dics

    def create_file_paths(self):
        """Create all directories', subdirectories', and files' paths ."""

        if self.job_dir is None:
            self.job_dir = os.path.basename(self.input_file).split(".")[0]

        if self.job_id is None:
            self.job_id = self.job_dir

        self.job_dir = os.path.join(self.simulations_dir, self.job_dir)

        # Create Processes directories
        self.processes_dir = [
            os.path.join(self.job_dir, self.preprocessing_dir),
            os.path.join(self.job_dir, self.simulation_dir),
            os.path.join(self.job_dir, self.postprocessing_dir),
        ]

        # Redundancy
        self.preprocessing_dir = self.processes_dir[0]
        self.simulation_dir = self.processes_dir[1]
        self.postprocessing_dir = self.processes_dir[2]

        # Redirect to the correct process folder
        if self.process == "preprocessing":
            indx = 0
        else:
            # Note that Postprocessing needs the link to simulation's folder
            # because that is where I look for energy files and pickle files
            indx = 1

        # Equilibration directory and sub_dir
        self.equilibration_dir = os.path.join(self.processes_dir[indx], self.equilibration_dir)
        self.eq_dump_dir = os.path.join(self.equilibration_dir, "dumps")
        # Production dir and sub_dir
        self.production_dir = os.path.join(self.processes_dir[indx], self.production_dir)
        self.prod_dump_dir = os.path.join(self.production_dir, "dumps")

        # Production phase filenames
        self.prod_energy_filename = os.path.join(self.production_dir, "ProductionEnergy_" + self.job_id + ".csv")
        self.prod_ptcls_filename = os.path.join(self.prod_dump_dir, "checkpoint_")

        # Equilibration phase filenames
        self.eq_energy_filename = os.path.join(self.equilibration_dir, "EquilibrationEnergy_" + self.job_id + ".csv")
        self.eq_ptcls_filename = os.path.join(self.eq_dump_dir, "checkpoint_")

        # Magnetic dir
        if self.electrostatic_equilibration:
            self.magnetization_dir = os.path.join(self.processes_dir[indx], self.magnetization_dir)
            self.mag_dump_dir = os.path.join(self.magnetization_dir, "dumps")
            # Magnetization phase filenames
            self.mag_energy_filename = os.path.join(self.magnetization_dir, "MagnetizationEnergy_" + self.job_id + ".csv")
            self.mag_ptcls_filename = os.path.join(self.mag_dump_dir, "checkpoint_")

        if self.process == "postprocessing":
            indx = 2  # Redirect to the correct folder

        # Log File
        if self.log_file is None:
            self.log_file = os.path.join(self.processes_dir[indx], "log_" + self.job_id + ".out")
        else:
            self.log_file = os.path.join(self.processes_dir[indx], self.log_file)

    def make_directories(self):
        """Create directories if non-existent."""

        # Check if the directories exist
        if not os.path.exists(self.simulations_dir):
            os.mkdir(self.simulations_dir)

        if not os.path.exists(self.job_dir):
            os.mkdir(self.job_dir)

        # Create Process' directories and their subdir
        for i in self.processes_dir:
            if not os.path.exists(i):
                os.mkdir(i)
        # The following automatically create directories in the correct Process
        if not os.path.exists(self.equilibration_dir):
            os.mkdir(self.equilibration_dir)

        if not os.path.exists(self.eq_dump_dir):
            os.mkdir(self.eq_dump_dir)

        if not os.path.exists(self.production_dir):
            os.mkdir(self.production_dir)

        if not os.path.exists(self.prod_dump_dir):
            os.mkdir(self.prod_dump_dir)

        if self.electrostatic_equilibration:
            if not os.path.exists(self.magnetization_dir):
                os.mkdir(self.magnetization_dir)

            if not os.path.exists(self.mag_dump_dir):
                os.mkdir(self.mag_dump_dir)

        if self.preprocessing:
            if not os.path.exists(self.preprocessing_dir):
                os.mkdir(self.preprocessing_dir)

        if not os.path.exists(self.postprocessing_dir):
            os.mkdir(self.postprocessing_dir)

    def file_header(self):
        """Create the log file and print the figlet if not a restart run."""

        if not self.restart:
            with open(self.log_file, "w+") as f_log:
                figlet_obj = Figlet(font="starwars")
                print(figlet_obj.renderText("Sarkas"), file=f_log)
                print("An open-source pure-Python molecular dynamics suite for non-ideal plasmas.", file=f_log)

        # Print figlet to screen if verbose
        if self.verbose:
            self.screen_figlet()

    def simulation_summary(self, simulation):
        """
        Print out to file a summary of simulation's parameters.
        If verbose output then it will print twice: the first time to file and second time to screen.

        Parameters
        ----------
        simulation : sarkas.processes.Process
            Simulation's parameters

        """

        screen = sys.stdout
        f_log = open(self.log_file, "a+")

        repeat = 2 if self.verbose else 1
        # redirect printing to file
        sys.stdout = f_log

        # Print to file first then to screen if repeat == 2
        while repeat > 0:

            if simulation.parameters.load_method in ["production_restart", "prod_restart"]:
                print("\n\n--------------------------- Production Restart -------------------------------------")
                self.time_info(simulation)
            elif simulation.parameters.load_method in ["equilibration_restart", "eq_restart"]:
                print("\n\n------------------------ Equilibration Restart ----------------------------------")
                self.time_info(simulation)
            elif simulation.parameters.load_method in ["magnetization_restart", "mag_restart"]:
                print("\n\n------------------------ Magnetization Restart ----------------------------------")
                self.time_info(simulation)
            elif self.process == "postprocessing":
                # Header of process
                process_title = "{:^80}".format(self.process.capitalize())
                print("\n\n")
                print(*["*" for i in range(50)])
                print(process_title)
                print(*["*" for i in range(50)])

                print("\nJob ID: ", self.job_id)
                print("Job directory: ", self.job_dir)
                print("PostProcessing directory: \n", self.postprocessing_dir)

                print("\nEquilibration dumps directory: ", self.eq_dump_dir)
                print("Production dumps directory: \n", self.prod_dump_dir)

                print("\nEquilibration Thermodynamics file: \n", self.eq_energy_filename)
                print("Production Thermodynamics file: \n", self.prod_energy_filename)

            else:

                # Header of process
                process_title = "{:^80}".format(self.process.capitalize())
                print("\n\n")
                print(*["*" for i in range(50)])
                print(process_title)
                print(*["*" for i in range(50)])

                print("\nJob ID: ", self.job_id)
                print("Job directory: ", self.job_dir)
                print("\nEquilibration dumps directory: \n", self.eq_dump_dir)
                print("Production dumps directory: \n", self.prod_dump_dir)

                print("\nEquilibration Thermodynamics file: \n", self.eq_energy_filename)
                print("Production Thermodynamics file: \n", self.prod_energy_filename)

                if hasattr(simulation.parameters, "rand_seed"):
                    print("Random Seed = ", simulation.parameters.rand_seed)

                print("\nPARTICLES:")
                print("Total No. of particles = ", simulation.parameters.total_num_ptcls)
                print("No. of species = ", len(simulation.species))
                for isp, sp in enumerate(simulation.species):
                    print("Species ID: {}".format(isp))
                    sp.pretty_print(simulation.potential.type, simulation.parameters.units)

                simulation.parameters.pretty_print()

                print("\nPOTENTIAL: ", simulation.potential.type)
                self.potential_info(simulation)

                print("\nALGORITHM: ", simulation.potential.method)
                self.algorithm_info(simulation)

                print("\nTHERMOSTAT: ")
                simulation.thermostat.pretty_print()

                print("\nINTEGRATOR: ")
                simulation.integrator.pretty_print(
                    simulation.parameters.total_plasma_frequency,
                    simulation.parameters.load_method,
                    simulation.parameters.restart_step,
                )

            repeat -= 1
            sys.stdout = screen  # Restore the original sys.stdout

        f_log.close()

    def time_stamp(self, time_stamp, t):
        """
        Print out to screen elapsed times. If verbose output, print to file first and then to screen.

        Parameters
        ----------
        time_stamp : str
            Array of time stamps.

        t : float
            Elapsed time.
        """
        screen = sys.stdout
        f_log = open(self.log_file, "a+")
        repeat = 2 if self.verbose else 1
        t_hrs, t_min, t_sec, t_msec, t_usec, t_nsec = t
        # redirect printing to file
        sys.stdout = f_log

        while repeat > 0:
            if "Potential Initialization" in time_stamp:
                print("\n\n{:-^70} \n".format(" Initialization Times "))
            if t_hrs == 0 and t_min == 0 and t_sec <= 2:
                print(
                    "\n{} Time: {} sec {} msec {} usec {} nsec".format(
                        time_stamp, int(t_sec), int(t_msec), int(t_usec), int(t_nsec)
                    )
                )
            else:
                print("\n{} Time: {} hrs {} min {} sec".format(time_stamp, int(t_hrs), int(t_min), int(t_sec)))

            repeat -= 1
            sys.stdout = screen

        f_log.close()

    def timing_study(self, simulation):
        """
        Info specific for timing study.

        Parameters
        ----------
        simulation: sarkas.core.Simulation
            Simulation's parameters

        """
        screen = sys.stdout
        f_log = open(self.log_file, "a+")
        repeat = 2 if self.verbose else 1

        # redirect printing to file
        sys.stdout = f_log

        # Print to file first then to screen if repeat == 2
        while repeat > 0:
            print("\n\n------------ Conclusion ------------\n")
            print("Suggested Mesh = [ {} , {} , {} ]".format(*simulation.potential.pppm_mesh))
            print(
                "Suggested Ewald parameter alpha = {:2.4f} / a_ws = {:1.6e} ".format(
                    simulation.potential.pppm_alpha_ewald * simulation.parameters.a_ws,
                    simulation.potential.pppm_alpha_ewald,
                ),
                end="",
            )
            print("[1/cm]" if simulation.parameters.units == "cgs" else "[1/m]")
            print(
                "Suggested rcut = {:2.4f} a_ws = {:.6e} ".format(
                    simulation.potential.rc / simulation.parameters.a_ws, simulation.potential.rc
                ),
                end="",
            )
            print("[cm]" if simulation.parameters.units == "cgs" else "[m]")

            self.algorithm_info(simulation)
            repeat -= 1
            sys.stdout = screen  # Restore the original sys.stdout

        f_log.close()

    def preprocess_sizing(self, sizes):
        """Print the estimated file sizes. """

        screen = sys.stdout
        f_log = open(self.log_file, "a+")
        repeat = 2 if self.verbose else 1

        # redirect printing to file
        sys.stdout = f_log
        while repeat > 0:
            print("\n\n{:=^70} \n".format(" Filesize Estimates "))
            size_GB, size_MB, size_KB, rem = convert_bytes(sizes[0, 0])
            print("\nEquilibration:\n")
            print(
                "Checkpoint filesize: {} GB {} MB {} KB {} bytes".format(
                    int(size_GB), int(size_MB), int(size_KB), int(rem)
                )
            )

            size_GB, size_MB, size_KB, rem = convert_bytes(sizes[0, 1])
            print(
                "Checkpoint folder size: {} GB {} MB {} KB {} bytes".format(
                    int(size_GB), int(size_MB), int(size_KB), int(rem)
                )
            )
            if self.electrostatic_equilibration:
                print("\nMagnetization:\n")
                size_GB, size_MB, size_KB, rem = convert_bytes(sizes[2, 0])
                print(
                    "Checkpoint filesize: {} GB {} MB {} KB {} bytes".format(
                        int(size_GB), int(size_MB), int(size_KB), int(rem)
                    )
                )

                size_GB, size_MB, size_KB, rem = convert_bytes(sizes[2, 1])
                print(
                    "Checkpoint folder size: {} GB {} MB {} KB {} bytes".format(
                        int(size_GB), int(size_MB), int(size_KB), int(rem)
                    )
                )

            size_GB, size_MB, size_KB, rem = convert_bytes(sizes[1, 0])
            print("\nProduction:\n")
            print(
                "Checkpoint filesize: {} GB {} MB {} KB {} bytes".format(
                    int(size_GB), int(size_MB), int(size_KB), int(rem)
                )
            )

            size_GB, size_MB, size_KB, rem = convert_bytes(sizes[1, 1])
            print(
                "Checkpoint folder size: {} GB {} MB {} KB {} bytes".format(
                    int(size_GB), int(size_MB), int(size_KB), int(rem)
                )
            )

            size_GB, size_MB, size_KB, rem = convert_bytes(sizes[:, 1].sum())
            print(
                "\nTotal minimum needed space: {} GB {} MB {} KB {} bytes".format(
                    int(size_GB), int(size_MB), int(size_KB), int(rem)
                )
            )
            repeat -= 1
            sys.stdout = screen

        f_log.close()

    def preprocess_timing(self, str_id, t, loops):
        """Print times estimates of simulation to file first and then to screen if verbose."""
        screen = sys.stdout
        f_log = open(self.log_file, "a+")
        repeat = 2 if self.verbose else 1
        t_hrs, t_min, t_sec, t_msec, t_usec, t_nsec = t
        # redirect printing to file
        sys.stdout = f_log
        while repeat > 0:
            if str_id == "header":
                print("\n\n{:=^70} \n".format(" Times Estimates "))
            elif str_id == "GF":
                print(
                    "Optimal Green's Function Time: \n"
                    "{} min {} sec {} msec {} usec {} nsec \n".format(
                        int(t_min), int(t_sec), int(t_msec), int(t_usec), int(t_nsec)
                    )
                )

            elif str_id == "PP":
                print(
                    "Time of PP acceleration calculation averaged over {} steps: \n"
                    "{} min {} sec {} msec {} usec {} nsec \n".format(
                        loops - 1, int(t_min), int(t_sec), int(t_msec), int(t_usec), int(t_nsec)
                    )
                )

            elif str_id == "PM":
                print(
                    "Time of PM acceleration calculation averaged over {} steps: \n"
                    "{} min {} sec {} msec {} usec {} nsec \n".format(
                        loops - 1, int(t_min), int(t_sec), int(t_msec), int(t_usec), int(t_nsec)
                    )
                )

            elif str_id == "Equilibration":
                print(
                    "Time of a single equilibration step averaged over {} steps: \n"
                    "{} min {} sec {} msec {} usec {} nsec \n".format(
                        loops - 1, int(t_min), int(t_sec), int(t_msec), int(t_usec), int(t_nsec)
                    )
                )
            elif str_id == "Magnetization":
                print(
                    "Time of a single magnetization step averaged over {} steps: \n"
                    "{} min {} sec {} msec {} usec {} nsec \n".format(
                        loops - 1, int(t_min), int(t_sec), int(t_msec), int(t_usec), int(t_nsec)
                    )
                )
            elif str_id == "Production":
                print(
                    "Time of a single production step averaged over {} steps: \n"
                    "{} min {} sec {} msec {} usec {} nsec \n".format(
                        loops - 1, int(t_min), int(t_sec), int(t_msec), int(t_usec), int(t_nsec)
                    )
                )

                print("\n\n{:-^70} \n".format(" Total Estimated Times "))
            repeat -= 1
            sys.stdout = screen

        f_log.close()

    def postprocess_info(self, simulation, write_to_file=False, observable=None):
        """
        Print Post-processing info to file and/or screen in a reader-friendly format.

        Parameters
        ----------
        simulation : sarkas.processes.PostProcess
            Sarkas processing stage.

        write_to_file : bool
            Flag for printing info also to file. Default= False.

        observable : str
            Observable whose info to print. Default = None.
            Choices = ['header','rdf', 'ccf', 'dsf', 'ssf', 'vm']

        """

        choices = ["header", "rdf", "ccf", "dsf", "ssf", "vd"]
        msg = (
            "Observable not defined. \n "
            "Please choose an observable from this list \n"
            "'rdf' = Radial Distribution Function, \n"
            "'ccf' = Current Correlation Function, \n"
            "'dsf' = Dynamic Structure Function, \n"
            "'ssf' = Static Structure Factor, \n"
            "'vd' = Velocity Distribution"
        )
        if observable is None:
            raise ValueError(msg)

        if observable not in choices:
            raise ValueError(msg)

        if write_to_file:
            screen = sys.stdout
            f_log = open(self.log_file, "a+")
            repeat = 2 if self.verbose else 1

            # redirect printing to file
            sys.stdout = f_log
        else:
            repeat = 1

        while repeat > 0:
            if observable == "header":
                # Header of process
                process_title = "{:^80}".format(self.process.capitalize())
                print("\n\n")
                print(*["*" for i in range(50)])
                print(process_title)
                print(*["*" for i in range(50)])

            elif observable == "rdf":
                simulation.rdf.pretty_print()
            elif observable == "ssf":
                simulation.ssf.pretty_print()
            elif observable == "dsf":
                simulation.dsf.pretty_print()
            elif observable == "ccf":
                simulation.ccf.pretty_print()
            elif observable == "vd":
                simulation.vm.setup(simulation.parameters)
                print("\nVelocity Moments:")
                print("Maximum no. of moments = {}".format(simulation.vm.max_no_moment))
                print("Maximum velocity moment = {}".format(int(2 * simulation.vm.max_no_moment)))

            repeat -= 1

            if write_to_file:
                sys.stdout = screen

        if write_to_file:
            f_log.close()

    @staticmethod
    def screen_figlet():
        """
        Print a colored figlet of Sarkas to screen.
        """
        if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
            # Assume white background in Jupyter Notebook
            clr = DARK_COLORS[np.random.randint(0, len(DARK_COLORS))]
        else:
            # Assume dark background in IPython/Python Kernel
            clr = LIGHT_COLORS[np.random.randint(0, len(LIGHT_COLORS))]
        fnt = FONTS[np.random.randint(0, len(FONTS))]
        print_figlet("\nSarkas\n", font=fnt, colors=clr)

        print("\nAn open-source pure-python molecular dynamics suite for non-ideal plasmas.\n\n")

    @staticmethod
    def time_info(simulation):
        """
        Print time simulation's parameters.

        Parameters
        ----------
        simulation: sarkas.core.Simulation
            Simulation's parameters.

        """
        wp_dt = simulation.parameters.total_plasma_frequency * simulation.integrator.dt
        print("Time step = {:.6e} [s]".format(simulation.integrator.dt))
        if simulation.potential.type in ["Yukawa", "EGS", "Coulomb", "Moliere"]:
            print("Total plasma frequency = {:1.6e} [rad/s]".format(simulation.parameters.total_plasma_frequency))
            print("w_p dt = {:2.4f}".format(wp_dt))
            if simulation.parameters.magnetized:
                if simulation.parameters.num_species > 1:
                    high_wc_dt = simulation.parameters.species_cyclotron_frequencies.max() * simulation.integrator.dt
                    low_wc_dt = simulation.parameters.species_cyclotron_frequencies.min() * simulation.integrator.dt
                    print("Highest w_c dt = {:2.4f}".format(high_wc_dt))
                    print("Smalles w_c dt = {:2.4f}".format(low_wc_dt))
                else:
                    high_wc_dt = simulation.parameters.species_cyclotron_frequencies.max() * simulation.integrator.dt
                    print("w_c dt = {:2.4f}".format(high_wc_dt))
        elif simulation.potential.type == "QSP":
            print("e plasma frequency = {:.6e} [rad/s]".format(simulation.species[0].plasma_frequency))
            print("ion plasma frequency = {:.6e} [rad/s]".format(simulation.species[1].plasma_frequency))
            print("w_pe dt = {:2.4f}".format(simulation.integrator.dt * simulation.species[0].plasma_frequency))
            if simulation.parameters.magnetized:
                if simulation.parameters.num_species > 1:
                    high_wc_dt = simulation.parameters.species_cyclotron_frequencies.max() * simulation.integrator.dt
                    low_wc_dt = simulation.parameters.species_cyclotron_frequencies.min() * simulation.integrator.dt
                    print("Electron w_ce dt = {:2.4f}".format(high_wc_dt))
                    print("Ions w_ci dt = {:2.4f}".format(low_wc_dt))
                else:
                    high_wc_dt = simulation.parameters.species_cyclotron_frequencies.max() * simulation.integrator.dt
                    print("w_c dt = {:2.4f}".format(high_wc_dt))
        elif simulation.potential.type == "LJ":
            print(
                "Total equivalent plasma frequency = {:1.6e} [rad/s]".format(simulation.parameters.total_plasma_frequency)
            )
            print("w_p dt = {:2.4f}".format(wp_dt))
            if simulation.parameters.magnetized:
                if simulation.parameters.num_species > 1:
                    high_wc_dt = simulation.parameters.species_cyclotron_frequencies.max() * simulation.integrator.dt
                    low_wc_dt = simulation.parameters.species_cyclotron_frequencies.min() * simulation.integrator.dt
                    print("Highest w_c dt = {:2.4f}".format(high_wc_dt))
                    print("Smalles w_c dt = {:2.4f}".format(low_wc_dt))
                else:
                    high_wc_dt = simulation.parameters.species_cyclotron_frequencies.max() * simulation.integrator.dt
                    print("w_c dt = {:2.4f}".format(high_wc_dt))

        # Print Time steps information
        # Check for restart simulations
        if simulation.parameters.load_method in ["production_restart", "prod_restart"]:
            print("Restart step: {}".format(simulation.parameters.restart_step))
            print(
                "Total production steps = {} \n"
                "Total production time = {:.4e} [s] ~ {} w_p T_prod ".format(
                    simulation.integrator.production_steps,
                    simulation.integrator.production_steps * simulation.integrator.dt,
                    int(simulation.integrator.production_steps * wp_dt),
                )
            )
            print(
                "snapshot interval step = {} \n"
                "snapshot interval time = {:.4e} [s] = {:.4f} w_p T_snap".format(
                    simulation.integrator.prod_dump_step,
                    simulation.integrator.prod_dump_step * simulation.integrator.dt,
                    simulation.integrator.prod_dump_step * wp_dt,
                )
            )

        elif simulation.parameters.load_method in ["equilibration_restart", "eq_restart"]:
            print("Restart step: {}".format(simulation.parameters.restart_step))
            print(
                "Total equilibration steps = {} \n"
                "Total equilibration time = {:.4e} [s] ~ {} w_p T_eq".format(
                    simulation.integrator.equilibration_steps,
                    simulation.integrator.equilibration_steps * simulation.integrator.dt,
                    int(simulation.integrator.eq_dump_step * wp_dt),
                )
            )
            print(
                "snapshot interval step = {} \n"
                "snapshot interval time = {:.4e} [s] = {:.4f} w_p T_snap".format(
                    simulation.integrator.eq_dump_step,
                    simulation.integrator.eq_dump_step * simulation.integrator.dt,
                    simulation.integrator.eq_dump_step * wp_dt,
                )
            )

        elif simulation.parameters.load_method in ["magnetization_restart", "mag_restart"]:
            print("Restart step: {}".format(simulation.parameters.restart_step))
            print(
                "Total magnetization steps = {} \n"
                "Total magnetization time = {:.4e} [s] ~ {} w_p T_mag".format(
                    simulation.integrator.magnetization_steps,
                    simulation.integrator.magnetization_steps * simulation.integrator.dt,
                    int(simulation.integrator.mag_dump_step * wp_dt),
                )
            )
            print(
                "snapshot interval step = {} \n"
                "snapshot interval time = {:.4e} [s] ~ {:1.4f} w_p T_snap".format(
                    simulation.integrator.mag_dump_step,
                    simulation.integrator.mag_dump_step * simulation.integrator.dt,
                    simulation.integrator.mag_dump_step * wp_dt,
                )
            )
        else:
            # Equilibration
            print(
                "\nEquilibration: \nNo. of equilibration steps = {} \n"
                "Total equilibration time = {:.4e} [s] ~ {} w_p T_eq ".format(
                    simulation.integrator.equilibration_steps,
                    simulation.integrator.equilibration_steps * simulation.integrator.dt,
                    int(simulation.integrator.equilibration_steps * wp_dt),
                )
            )
            print(
                "snapshot interval step = {} \n"
                "snapshot interval time = {:.4e} [s] = {:.4f} w_p T_snap".format(
                    simulation.integrator.eq_dump_step,
                    simulation.integrator.eq_dump_step * simulation.integrator.dt,
                    simulation.integrator.eq_dump_step * wp_dt,
                )
            )
            # Magnetization
            if simulation.integrator.electrostatic_equilibration:
                print(
                    "\nMagnetization: \nNo. of magnetization steps = {} \n"
                    "Total magnetization time = {:.4e} [s] ~ {} w_p T_mag ".format(
                        simulation.integrator.magnetization_steps,
                        simulation.integrator.magnetization_stepss * simulation.integrator.dt,
                        int(simulation.integrator.magnetization_steps * wp_dt),
                    )
                )

                print(
                    "snapshot interval step = {} \n"
                    "snapshot interval time = {:.4e} [s] = {:.4f} w_p T_snap".format(
                        simulation.integrator.mag_dump_step,
                        simulation.integrator.mag_dump_step * simulation.integrator.dt,
                        simulation.integrator.mag_dump_step * wp_dt,
                    )
                )
            # Production
            print(
                "\nProduction: \nNo. of production steps = {} \n"
                "Total production time = {:.4e} [s] ~ {} w_p T_prod ".format(
                    simulation.integrator.production_steps,
                    simulation.integrator.production_steps * simulation.integrator.dt,
                    int(simulation.integrator.production_steps * wp_dt),
                )
            )
            print(
                "snapshot interval step = {} \n"
                "snapshot interval time = {:.4e} [s] = {:.4f} w_p T_snap".format(
                    simulation.integrator.prod_dump_step,
                    simulation.integrator.prod_dump_step * simulation.integrator.dt,
                    simulation.integrator.prod_dump_step * wp_dt,
                )
            )

    @staticmethod
    def algorithm_info(simulation):
        """
        Print algorithm information.

        Parameters
        ----------
        simulation: sarkas.core.Simulation
            Simulation's parameters.


        """
        if simulation.potential.method == "pppm":
            print("Charge assignment order: {}".format(simulation.potential.pppm_cao))
            print("FFT aliases: [{}, {}, {}]".format(*simulation.potential.pppm_aliases))
            print("Mesh: {} x {} x {}".format(*simulation.potential.pppm_mesh))
            print(
                "Ewald parameter alpha = {:2.4f} / a_ws = {:1.6e} ".format(
                    simulation.potential.pppm_alpha_ewald * simulation.parameters.a_ws,
                    simulation.potential.pppm_alpha_ewald,
                ),
                end="",
            )
            print("[1/cm]" if simulation.parameters.units == "cgs" else "[1/m]")
            print(
                "Mesh width = {:.4f}, {:.4f}, {:.4f} a_ws".format(
                    simulation.potential.pppm_h_array[0] / simulation.parameters.a_ws,
                    simulation.potential.pppm_h_array[1] / simulation.parameters.a_ws,
                    simulation.potential.pppm_h_array[2] / simulation.parameters.a_ws,
                )
            )
            print(
                "           = {:.4e}, {:.4e}, {:.4e} ".format(
                    simulation.potential.pppm_h_array[0],
                    simulation.potential.pppm_h_array[1],
                    simulation.potential.pppm_h_array[2],
                ),
                end="",
            )
            print("[cm]" if simulation.parameters.units == "cgs" else "[m]")
            print(
                "Mesh size * Ewald_parameter (h * alpha) = {:2.4f}, {:2.4f}, {:2.4f} ".format(
                    simulation.potential.pppm_h_array[0] * simulation.potential.pppm_alpha_ewald,
                    simulation.potential.pppm_h_array[1] * simulation.potential.pppm_alpha_ewald,
                    simulation.potential.pppm_h_array[2] * simulation.potential.pppm_alpha_ewald,
                )
            )
            print(
                "                                        ~ 1/{}, 1/{}, 1/{}".format(
                    int(1.0 / (simulation.potential.pppm_h_array[0] * simulation.potential.pppm_alpha_ewald)),
                    int(1.0 / (simulation.potential.pppm_h_array[1] * simulation.potential.pppm_alpha_ewald)),
                    int(1.0 / (simulation.potential.pppm_h_array[2] * simulation.potential.pppm_alpha_ewald)),
                )
            )
            print(
                "rcut = {:2.4f} a_ws = {:.6e} ".format(
                    simulation.potential.rc / simulation.parameters.a_ws, simulation.potential.rc
                ),
                end="",
            )
            print("[cm]" if simulation.parameters.units == "cgs" else "[m]")
            print(
                "No. of PP cells per dimension = {:2}, {:2}, {:2}".format(
                    int(simulation.parameters.box_lengths[0] / simulation.potential.rc),
                    int(simulation.parameters.box_lengths[1] / simulation.potential.rc),
                    int(simulation.parameters.box_lengths[2] / simulation.potential.rc),
                )
            )
            print(
                "No. of particles in PP loop = {:6}".format(
                    int(
                        simulation.parameters.total_num_ptcls
                        / simulation.parameters.box_volume
                        * (3 * simulation.potential.rc) ** 3
                    )
                )
            )
            print(
                "No. of PP neighbors per particle = {:6}".format(
                    int(
                        simulation.parameters.total_num_ptcls
                        / simulation.parameters.box_volume
                        * 4.0
                        / 3.0
                        * np.pi
                        * (simulation.potential.rc) ** 3.0
                    )
                )
            )
            print("PM Force Error = {:.6e}".format(simulation.parameters.pppm_pm_err))
            print("PP Force Error = {:.6e}".format(simulation.parameters.pppm_pp_err))

        elif simulation.potential.method == "pp":
            print(
                "rcut = {:2.4f} a_ws = {:.6e} ".format(
                    simulation.potential.rc / simulation.parameters.a_ws, simulation.potential.rc
                ),
                end="",
            )
            print("[cm]" if simulation.parameters.units == "cgs" else "[m]")
            print(
                "No. of PP cells per dimension = {:2}, {:2}, {:2}".format(
                    int(simulation.parameters.box_lengths[0] / simulation.potential.rc),
                    int(simulation.parameters.box_lengths[1] / simulation.potential.rc),
                    int(simulation.parameters.box_lengths[2] / simulation.potential.rc),
                )
            )
            print(
                "No. of particles in PP loop = {:6}".format(
                    int(
                        simulation.parameters.total_num_ptcls
                        / simulation.parameters.box_volume
                        * (3 * simulation.potential.rc) ** 3
                    )
                )
            )
            print(
                "No. of PP neighbors per particle = {:6}".format(
                    int(
                        simulation.parameters.total_num_ptcls
                        * 4.0
                        / 3.0
                        * np.pi
                        * (simulation.potential.rc / simulation.parameters.box_lengths.min()) ** 3.0
                    )
                )
            )

        print("Tot Force Error = {:.6e}".format(simulation.parameters.force_error))

    @staticmethod
    def potential_info(simulation):
        """
        Print potential information.

        Parameters
        ----------
        simulation: sarkas.core.Simulation
            Simulation's parameters.

        """
        if simulation.potential.type == "yukawa":
            print(
                "electron temperature = {:1.4e} [K] = {:1.4e} [eV]".format(
                    simulation.parameters.electron_temperature,
                    simulation.parameters.electron_temperature / simulation.parameters.eV2K,
                )
            )
            print("kappa = {:.4f}".format(simulation.parameters.a_ws / simulation.parameters.lambda_TF))
            print("Gamma_eff = {:.2f}".format(simulation.parameters.coupling_constant))

        elif simulation.potential.type == "egs":
            # print('electron temperature = {:1.4e} [K] = {:1.4e} eV'.format(
            #     simulation.parameters.electron_temperature,
            #     simulation.parameters.electron_temperature / simulation.parameters.eV2K))
            print("kappa = {:.4f}".format(simulation.parameters.a_ws / simulation.parameters.lambda_TF))
            print("SGA Correction factor: lmbda = {:.4f}".format(simulation.potential.lmbda))
            # print('lambda_TF = {:1.4e} '.format(simulation.parameters.lambda_TF), end='')
            # print("[cm]" if simulation.parameters.units == "cgs" else "[m]")
            print("nu = {:.4f}".format(simulation.parameters.nu))
            if simulation.parameters.nu < 1:
                print("Exponential decay:")
                print("lambda_p = {:.6e} ".format(simulation.parameters.lambda_p), end="")
                print("[cm]" if simulation.parameters.units == "cgs" else "[m]")
                print("lambda_m = {:.6e} ".format(simulation.parameters.lambda_m), end="")
                print("[cm]" if simulation.parameters.units == "cgs" else "[m]")
                print("alpha = {:.4f}".format(simulation.parameters.alpha))
                # print('Theta = {:1.4e}'.format(simulation.parameters.electron_degeneracy_parameter))
                print("b = {:.4f}".format(simulation.parameters.b))

            else:
                print("Oscillatory potential:")
                print("gamma_p = {:.6e} ".format(simulation.parameters.gamma_p), end="")
                print("[cm]" if simulation.parameters.units == "cgs" else "[m]")
                print("gamma_m = {:.6e} ".format(simulation.parameters.gamma_m), end="")
                print("[cm]" if simulation.parameters.units == "cgs" else "[m]")
                print("alpha = {:.4f}".format(simulation.parameters.alphap))
                print("b = {:.4f}".format(simulation.parameters.b))

            print("Gamma_eff = {:4.2f}".format(simulation.parameters.coupling_constant))

        elif simulation.potential.type == "coulomb":
            print("Effective Coupling constant: Gamma_eff = {:4.2f}".format(simulation.parameters.coupling_constant))
            print("Short-range Cutoff radius: a_rs = {:.6e} ".format(simulation.potential.a_rs), end="")
            print("[cm]" if simulation.parameters.units == "cgs" else "[m]")
            # simulation.parameters.pretty_print()

        elif simulation.potential.type == "lj":
            print("epsilon = {:.6e}".format(simulation.potential.matrix[0, 0, 0]))
            print("sigma = {:.6e}".format(simulation.potential.matrix[1, 0, 0]))
            print("Gamma_eff = {:.2f}".format(simulation.parameters.coupling_constant))

        elif simulation.potential.type == "qsp":
            print("QSP type: {}".format(simulation.potential.qsp_type))
            print("Pauli term: {}".format(simulation.potential.qsp_pauli))
            print(
                "e de Broglie wavelength = {:.4f} a_ws = {:.6e} ".format(
                    np.sqrt(2.0) * np.pi / (simulation.potential.matrix[1, 0, 0] * simulation.parameters.a_ws),
                    np.sqrt(2.0) * np.pi / simulation.potential.matrix[1, 0, 0],
                ),
                end="",
            )
            print("[cm]" if simulation.parameters.units == "cgs" else "[m]")
            print(
                "ion de Broglie wavelength  = {:.4f} a_ws = {:.6e} ".format(
                    np.sqrt(2.0) * np.pi / (simulation.potential.matrix[1, 1, 1] * simulation.parameters.a_ws),
                    np.sqrt(2.0) * np.pi / simulation.potential.matrix[1, 1, 1],
                ),
                end="",
            )
            print("[cm]" if simulation.parameters.units == "cgs" else "[m]")
            print(
                "e-e screening length = {:.4f} a_ws = {:.6e} ".format(
                    1.0 / (simulation.potential.matrix[1, 0, 0] * simulation.parameters.a_ws),
                    1.0 / simulation.potential.matrix[1, 0, 0],
                ),
                end="",
            )
            print("[cm]" if simulation.parameters.units == "cgs" else "[m]")
            print(
                "i-i screening length = {:.4f} a_ws = {:.6e} ".format(
                    1.0 / (simulation.potential.matrix[1, 1, 1] * simulation.parameters.a_ws),
                    1.0 / simulation.potential.matrix[1, 1, 1],
                ),
                end="",
            )
            print("[cm]" if simulation.parameters.units == "cgs" else "[m]")
            print(
                "e-i screening length = {:.4f} a_ws = {:.6e} ".format(
                    1.0 / (simulation.potential.matrix[1, 0, 1] * simulation.parameters.a_ws),
                    1.0 / simulation.potential.matrix[1, 0, 1],
                ),
                end="",
            )
            print("[cm]" if simulation.parameters.units == "cgs" else "[m]")
            print("e-i coupling constant = {:.4f} ".format(simulation.parameters.coupling_constant))

        elif simulation.potential.type == "hs_yukawa":
            print(
                "electron temperature = {:1.4e} [K] = {:1.4e} [eV]".format(
                    simulation.parameters.electron_temperature,
                    simulation.parameters.electron_temperature / simulation.parameters.eV2K,
                )
            )
            print("kappa = {:.4f}".format(simulation.parameters.a_ws / simulation.parameters.lambda_TF))
            print(
                "short range cutoff = {:.2f} a_ws = {:.4e} ".format(
                    simulation.potential.matrix[-1, 0, 0] / simulation.parameters.a_ws,
                    simulation.potential.matrix[-1, 0, 0],
                ),
                end="",
            )
            print("[cm]" if simulation.parameters.units == "cgs" else "[m]")

            print("Gamma_eff = {:.2f}".format(simulation.parameters.coupling_constant))

    def setup_checkpoint(self, params, species):
        """
        Assign attributes needed for saving dumps.

        Parameters
        ----------
        params: sarkas.core.Parameters
            General simulation parameters.

        species: sarkas.core.Species
            List of Species classes.

        """
        self.dt = params.dt
        self.a_ws = params.a_ws
        self.total_num_ptcls = params.total_num_ptcls
        self.total_plasma_frequency = params.total_plasma_frequency
        self.species_names = np.copy(params.species_names)
        self.coupling = params.coupling_constant * params.T_desired

        # Check whether energy files exist already
        if not os.path.exists(self.prod_energy_filename):
            # Create the Energy file
            dkeys = ["Time", "Total Energy", "Total Kinetic Energy", "Potential Energy", "Temperature"]
            if len(species) > 1:
                for i, sp in enumerate(species):
                    dkeys.append("{} Kinetic Energy".format(sp.name))
                    dkeys.append("{} Potential Energy".format(sp.name))
                    dkeys.append("{} Temperature".format(sp.name))
            data = dict.fromkeys(dkeys)

            with open(self.prod_energy_filename, "w+") as f:
                w = csv.writer(f)
                w.writerow(data.keys())

        if not os.path.exists(self.eq_energy_filename) and not params.load_method[-7:] == "restart":
            # Create the Energy file
            dkeys = ["Time", "Total Energy", "Total Kinetic Energy", "Potential Energy", "Temperature"]
            if len(species) > 1:
                for i, sp_name in enumerate(params.species_names):
                    dkeys.append("{} Kinetic Energy".format(sp_name))
                    dkeys.append("{} Potential Energy".format(sp_name))
                    dkeys.append("{} Temperature".format(sp_name))
            data = dict.fromkeys(dkeys)

            with open(self.eq_energy_filename, "w+") as f:
                w = csv.writer(f)
                w.writerow(data.keys())

        if self.electrostatic_equilibration:
            if not os.path.exists(self.mag_energy_filename) and not params.load_method[-7:] == "restart":
                # Create the Energy file
                dkeys = ["Time", "Total Energy", "Total Kinetic Energy", "Potential Energy", "Temperature"]
                if len(species) > 1:
                    for i, sp_name in enumerate(params.species_names):
                        dkeys.append("{} Kinetic Energy".format(sp_name))
                        dkeys.append("{} Potential Energy".format(sp_name))
                        dkeys.append("{} Temperature".format(sp_name))
                data = dict.fromkeys(dkeys)

                with open(self.mag_energy_filename, "w+") as f:
                    w = csv.writer(f)
                    w.writerow(data.keys())

    def save_pickle(self, simulation):
        """
        Save all simulations parameters in pickle files.
        """
        file_list = ["parameters", "integrator", "thermostat", "potential", "species"]

        # Redirect to the correct process folder
        if self.process == "preprocessing":
            indx = 0
        else:
            # Note that Postprocessing needs the link to simulation's folder
            # because that is where I look for energy files and pickle files
            indx = 1

        for fl in file_list:
            filename = os.path.join(self.processes_dir[indx], fl + ".pickle")
            pickle_file = open(filename, "wb")
            pickle.dump(simulation.__dict__[fl], pickle_file)
            pickle_file.close()

    def read_pickle(self, process):
        """
        Read pickle files containing all the simulation information.

        Parameters
        ----------
        process: cls
            Simulation's parameters. It can be one of three (sarkas.tools.PreProcess,
            sarkas.core.Simulation, sarkas.tools.PostProcess)
        """
        import copy as py_copy

        file_list = ["parameters", "integrator", "thermostat", "potential", "species"]

        # Redirect to the correct process folder
        if self.process == "preprocessing":
            indx = 0
        else:
            # Note that Postprocessing needs the link to simulation's folder
            # because that is where I look for energy files and pickle files
            indx = 1

        for fl in file_list:
            filename = os.path.join(self.processes_dir[indx], fl + ".pickle")
            data = np.load(filename, allow_pickle=True)
            process.__dict__[fl] = py_copy.copy(data)

    def read_pickle_single(self, class_to_read):
        """
        Read the desired pickle file.

        Parameters
        ----------
        class_to_read : str
            Name of the class to read.

        Returns
        -------
        : cls
            Desired class.

        """
        import copy as py_copy

        # Redirect to the correct process folder
        if self.process == "preprocessing":
            indx = 0
        else:
            # Note that Postprocessing needs the link to simulation's folder
            # because that is where I look for energy files and pickle files
            indx = 1

        filename = os.path.join(self.processes_dir[indx], class_to_read + ".pickle")
        data = np.load(filename, allow_pickle=True)
        return py_copy.copy(data)

    def dump(self, phase, ptcls, it):
        """
        Save particles' data to binary file for future restart.

        Parameters
        ----------
        phase: str
            Simulation phase.

        ptcls: sarkas.core.Particles
            Particles data.

        it : int
            Timestep number.
        """
        if phase == "production":
            ptcls_file = self.prod_ptcls_filename + str(it)
            tme = it * self.dt
            np.savez(
                ptcls_file,
                id=ptcls.id,
                names=ptcls.names,
                pos=ptcls.pos,
                vel=ptcls.vel,
                acc=ptcls.acc,
                cntr=ptcls.pbc_cntr,
                rdf_hist=ptcls.rdf_hist,
                virial=ptcls.virial,
                time=tme,
            )

            energy_file = self.prod_energy_filename

        elif phase == "equilibration":
            ptcls_file = self.eq_ptcls_filename + str(it)
            tme = it * self.dt
            np.savez(
                ptcls_file,
                id=ptcls.id,
                names=ptcls.names,
                pos=ptcls.pos,
                vel=ptcls.vel,
                acc=ptcls.acc,
                virial=ptcls.virial,
                time=tme,
            )

            energy_file = self.eq_energy_filename

        elif phase == "magnetization":
            ptcls_file = self.mag_ptcls_filename + str(it)
            tme = it * self.dt
            np.savez(
                ptcls_file,
                id=ptcls.id,
                names=ptcls.names,
                pos=ptcls.pos,
                vel=ptcls.vel,
                acc=ptcls.acc,
                virial=ptcls.virial,
                time=tme,
            )

            energy_file = self.mag_energy_filename

        kinetic_energies, temperatures = ptcls.kinetic_temperature()
        potential_energies = ptcls.potential_energies()
        # Save Energy data
        data = {
            "Time": it * self.dt,
            "Total Energy": np.sum(kinetic_energies) + ptcls.potential_energy,
            "Total Kinetic Energy": np.sum(kinetic_energies),
            "Potential Energy": ptcls.potential_energy,
            "Total Temperature": ptcls.species_num.transpose() @ temperatures / ptcls.total_num_ptcls,
        }
        if len(temperatures) > 1:
            for sp, kin in enumerate(kinetic_energies):
                data["{} Kinetic Energy".format(self.species_names[sp])] = kin
                data["{} Potential Energy".format(self.species_names[sp])] = potential_energies[sp]
                data["{} Temperature".format(self.species_names[sp])] = temperatures[sp]

        with open(energy_file, "a") as f:
            w = csv.writer(f)
            w.writerow(data.values())

    def dump_xyz(self, phase="production"):
        """
        Save the XYZ file by reading Sarkas dumps.

        Parameters
        ----------
        phase : str
            Phase from which to read dumps. 'equilibration' or 'production'.

        dump_skip : int
            Interval of dumps to skip. Default = 1

        """

        if phase == "equilibration":
            self.xyz_filename = os.path.join(self.equilibration_dir, "pva_" + self.job_id + ".xyz")
            dump_dir = self.eq_dump_dir

        else:
            self.xyz_filename = os.path.join(self.production_dir, "pva_" + self.job_id + ".xyz")
            dump_dir = self.prod_dump_dir

        f_xyz = open(self.xyz_filename, "w+")

        if not hasattr(self, "a_ws"):
            params = self.read_pickle_single("parameters")
            self.a_ws = params.a_ws
            self.total_num_ptcls = params.total_num_ptcls
            self.total_plasma_frequency = params.total_plasma_frequency

        # Rescale constants. This is needed since OVITO has a small number limit.
        pscale = 1.0 / self.a_ws
        vscale = 1.0 / (self.a_ws * self.total_plasma_frequency)
        ascale = 1.0 / (self.a_ws * self.total_plasma_frequency ** 2)

        # Read the list of dumps and sort them in the correct (natural) order
        dumps = os.listdir(dump_dir)
        dumps.sort(key=num_sort)
        for dump in tqdm(dumps, disable=not self.verbose):
            data = self.read_npz(dump_dir, dump)
            data["pos_x"] *= pscale
            data["pos_y"] *= pscale
            data["pos_z"] *= pscale

            data["vel_x"] *= vscale
            data["vel_y"] *= vscale
            data["vel_z"] *= vscale

            data["acc_x"] *= ascale
            data["acc_y"] *= ascale
            data["acc_z"] *= ascale

            f_xyz.writelines("{0:d}\n".format(self.total_num_ptcls))
            f_xyz.writelines("name x y z vx vy vz ax ay az\n")
            np.savetxt(f_xyz, data, fmt="%s %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e")

        f_xyz.close()

    @staticmethod
    def read_npz(fldr, it):
        """
        Load particles' data from dumps.

        Parameters
        ----------
        fldr : str
            Folder containing dumps.

        it : str
            Timestep to load.

        Returns
        -------
        struct_array : numpy.ndarray
            Structured data array.

        """

        file_name = os.path.join(fldr, it)
        data = np.load(file_name, allow_pickle=True)
        # Dev Notes: the old way of saving the xyz file by
        # np.savetxt(f_xyz, np.c_[data["names"],data["pos"] ....]
        # , fmt="%10s %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e")
        # was not working, because the columns of np.c_[] all have the same data type <U32
        # which is in conflict with the desired fmt. i.e. data["names"] was not recognized as a string.
        # So I have to create a new structured array and pass this. I could not think of a more Pythonic way.
        struct_array = np.zeros(
            data["names"].size,
            dtype=[
                ("names", "U6"),
                ("pos_x", np.float64),
                ("pos_y", np.float64),
                ("pos_z", np.float64),
                ("vel_x", np.float64),
                ("vel_y", np.float64),
                ("vel_z", np.float64),
                ("acc_x", np.float64),
                ("acc_y", np.float64),
                ("acc_z", np.float64),
            ],
        )
        struct_array["names"] = data["names"]
        struct_array["pos_x"] = data["pos"][:, 0]
        struct_array["pos_y"] = data["pos"][:, 1]
        struct_array["pos_z"] = data["pos"][:, 2]

        struct_array["vel_x"] = data["vel"][:, 0]
        struct_array["vel_y"] = data["vel"][:, 1]
        struct_array["vel_z"] = data["vel"][:, 2]

        struct_array["acc_x"] = data["acc"][:, 0]
        struct_array["acc_y"] = data["acc"][:, 1]
        struct_array["acc_z"] = data["acc"][:, 2]

        return struct_array


def alpha_to_int(text):
    return int(text) if text.isdigit() else text


def num_sort(text):
    """
    Method copied from
    https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside

    Notes
    -----
    originally from http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)

    Parameters
    ----------
    text : str
        Text to be split into str and int

    Returns
    -------
     : list
        List containing text and integers

    """

    return [alpha_to_int(c) for c in re.split(r"(\d+)", text)]


def convert_bytes(tot_bytes):
    GB, rem = divmod(tot_bytes, 1024 * 1024 * 1024)
    MB, rem = divmod(rem, 1024 * 1024)
    KB, rem = divmod(rem, 1024)

    return [GB, MB, KB, rem]
