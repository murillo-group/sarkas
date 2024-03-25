"""
Module handling the I/O for an MD run.
"""
import csv
import datetime
import pickle
import re
import sys
import yaml
from copy import copy, deepcopy
from IPython import get_ipython
from numpy import c_, float64
from numpy import load as np_load
from numpy import savetxt, savez, zeros
from numpy.random import randint
from os import listdir, mkdir, symlink
from os.path import basename, exists, join
from pyfiglet import Figlet, print_figlet
from warnings import warn

if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    # If you are using Jupyter Notebook
    from tqdm.notebook import trange
else:
    # If you are using IPython or Python kernel
    from tqdm import trange

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
    """
    Class handling the input and output functions of the MD run.

    Parameters
    ----------
    process : str
        Name of the process class containing MD run info.

    """

    def __init__(self, process: str = "preprocess"):

        self.electrostatic_equilibration: bool = True
        self.eq_dump_dir: str = "dumps"
        self.equilibration_dir: str = "Equilibration"
        self.input_file: str = None  # MD run input file.
        self.job_dir: str = None
        self.job_id: str = None
        self.log_file: str = None
        self.mag_dump_dir: str = "dumps"
        self.magnetization_dir: str = "Magnetization"
        self.magnetized: bool = False
        self.preprocess_file: str = None
        self.preprocessing: bool = False
        self.preprocessing_dir: str = "PreProcessing"
        self.prod_dump_dir: str = "dumps"
        self.production_dir: str = "Production"
        self.postprocessing_dir: str = "PostProcessing"
        self.md_simulations_dir: str = "SarkasSimulations"
        self.simulation_dir: str = "Simulation"

        self.process_names_dict = {
            "preprocessing": "PreProcessing",
            "simulation": "Simulation",
            "postprocessing": "PostProcessing",
        }
        self.phase_names_dict = {
            "equilibration": "Equilibration",
            "production": "Production",
            "magnetization": "Magnetization",
        }

        self.filenames_tree = {
            "log": {"path": "log", "name": "log.out"},
            "particles": {
                "equilibration": {"path": "dumps", "name": "checkpoint_"},
                "magnetization": {"path": "dumps", "name": "checkpoint_"},
                "production": {"path": "dumps", "name": "checkpoint_"},
            },
            "thermodynamics": {
                "equilibration": {"path": "process", "name": "energy.csv"},
                "magnetization": {"path": "process", "name": "energy.csv"},
                "production": {"path": "process", "name": "energy.csv"},
            },
        }

        self.particles_filepaths = {"equilibration": None, "magnetization": None, "production": None}
        self.thermodynamics_filepaths = {"equilibration": None, "magnetization": None, "production": None}

        self.thermodynamics_to_save = []  #: bool = False

        self.verbose: bool = False
        self.xyz_dir: str = None
        self.xyz_filename: str = None
        self.process = process
        self.data_to_save = ["pos", "vel", "acc", "rdf_hist"]

    def __repr__(self):
        sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
        disp = "InputOuput( \n"
        for key, value in sortedDict.items():
            disp += "\t{} : {}\n".format(key, value)
        disp += ")"
        return disp

    def __copy__(self):
        """Make a shallow copy of the object using copy by creating a new instance of the object and copying its __dict__."""
        # Create a new object
        _copy = type(self)()
        # copy the dictionary
        _copy.__dict__.update(self.__dict__)
        return _copy

    @staticmethod
    def algorithm_info(simulation):
        """
        Print algorithm information.

        Parameters
        ----------
        simulation : :class:`sarkas.processes.Process`
            Process class containing the algorithm info and other parameters.

        """
        warn(
            "Deprecated feature. It will be removed in a future release. Use potential.method_pretty_print()",
            category=DeprecationWarning,
        )
        simulation.potential.method_pretty_print()

    def copy_params(self, params):
        """
        Copy necessary parameters.

        Parameters
        ----------
        params: :class:`sarkas.core.Parameters`
            Simulation's parameters.

        """
        self.dt = params.dt
        self.a_ws = params.a_ws
        self.total_num_ptcls = params.total_num_ptcls
        self.total_plasma_frequency = params.total_plasma_frequency
        self.species_names = params.species_names.copy()
        self.coupling = params.coupling_constant * params.T_desired

        self.equilibration_phase = params.equilibration_phase

        self.eq_dump_step = params.eq_dump_step
        self.mag_dump_step = params.mag_dump_step
        self.prod_dump_step = params.prod_dump_step

        self.equilibration_steps = params.equilibration_steps
        self.magentization_steps = params.magnetization_steps
        self.production_steps = params.production_steps

    def create_file_paths(self):
        """Create all directories', subdirectories', and files' paths.

        Raises
        ------
            : DeprecationWarning
        """

        warn(
            "Deprecated feature. It will be removed in a future release.\nUse :meth:`make_files_tree`.",
            category=DeprecationWarning,
        )
        self.make_files_tree()

    def dump(self, phase, ptcls, it):
        """
        Save particles' position, velocity, and acceleration data to binary file for future restart.

        Parameters
        ----------
        phase : str
            Simulation phase.

        ptcls : :class:`sarkas.particles.Particles`
            Particles data.

        it : int
            Timestep number.
        """
        # if phase == "production":
        #     ptcls_file = self.prod_ptcls_filename + str(it)

        # elif phase == "equilibration":
        #     ptcls_file = self.eq_ptcls_filename + str(it)

        # elif phase == "magnetization":
        #     ptcls_file = self.mag_ptcls_filename + str(it)

        # self.dump_pva(phase, ptcls, it)
        self.dump_observables(phase, ptcls, it)
        self.dump_thermodynamics(phase, ptcls, it)

    def dump_observables(self, phase, ptcls, it):
        """
        Save particles' data to binary file (uncompressed npz) for future restart.

        Parameters
        ----------
        phase : str
            Simulation phase.

        ptcls : :class:`sarkas.particles.Particles`
            Particles data.

        it : int
            Timestep number.
        """

        tme = it * self.dt
        kwargs = {key: ptcls.__dict__[key] for key in self.data_to_save}
        kwargs["time"] = tme
        savez(f"{self.particles_filepaths[phase]}{it}", **kwargs)

    def dump_thermodynamics(self, phase, ptcls, it):
        """
        Save particles' data to binary file for future restart.

        Parameters
        ----------
        phase : str
            Simulation phase.

        ptcls : :class:`sarkas.particles.Particles`
            Particles data.

        it : int
            Timestep number.
        """

        data = {"Time": it * self.dt}
        datap = ptcls.make_thermodynamics_dictionary()
        data.update(datap)
        with open(self.thermodynamics_filepaths[phase], "a") as f:
            w = csv.writer(f)
            w.writerow(data.values())

    def dump_xyz(self, phase: str = "production", dump_start: int = 0, dump_end: int = None, dump_skip: int = 1):
        """
        Save the XYZ file by reading Sarkas dumps.

        Parameters
        ----------
        phase : str
            Phase from which to read dumps. 'equilibration' or 'production'.

        dump_start : int
            Step number from which to start saving. Default is 0.

        dump_end: int
            Last step number to save. Default is None.

        dump_skip : int
            Interval of dumps to skip. Default is 1

        """
        # TODO: Find a way so that the user can pass strings of the information to save for OVITO.
        # For example. the user might want to save pos, vel, acc and the kinetic energy or total energy of each particle.

        if phase == "equilibration":
            self.xyz_filename = join(self.equilibration_dir, "pva_" + self.job_id + ".xyz")
            dump_dir = self.eq_dump_dir
            dump_step = self.eq_dump_step

        elif phase == "magnetization":
            self.xyz_filename = join(self.magnetization_dir, "pva_" + self.job_id + ".xyz")
            dump_dir = self.mag_dump_dir
            dump_step = self.mag_dump_step

        else:
            self.xyz_filename = join(self.production_dir, "pva_" + self.job_id + ".xyz")
            dump_dir = self.prod_dump_dir
            dump_step = self.prod_dump_step

        # Get particles info
        params = self.read_pickle_single("parameters")

        # Rescale constants. This is needed since OVITO has a small number limit.
        pscale = 1.0 / params.a_ws
        vscale = 1.0 / (params.a_ws * params.total_plasma_frequency)
        ascale = 1.0 / (params.a_ws * params.total_plasma_frequency**2)

        f_xyz = open(self.xyz_filename, "w+")

        # Read the list of dumps and sort them in the correct (natural) order
        dumps = listdir(dump_dir)
        dumps.sort(key=num_sort)
        dumps_dict = {}
        for i in dumps:
            _, key = i.split("_")
            key_num, _ = key.split(".")
            dumps_dict[int(key_num)] = i

        if not dump_end:
            dump_end = len(dumps) * dump_step

        dump_skip *= dump_step

        for i in trange(dump_start, dump_end, dump_skip, disable=not self.verbose):
            dump = dumps_dict[i]
            
            # TODO: Do we really need to use read_particles_npz ? Can we find a way to save the .xyz file in a different way?
            # I need a structured array for np.c_[], 
            data = self.read_particles_npz(dump_dir, dump)
            data["pos_x"] *= pscale
            data["pos_y"] *= pscale
            data["pos_z"] *= pscale

            data["vel_x"] *= vscale
            data["vel_y"] *= vscale
            data["vel_z"] *= vscale
            # TODO: This rescale the acceleration. However, OVITO prefers the Force.
            # Find a way to generalize the force rescaling and save the forces instead of the accelerations
            data["acc_x"] *= ascale
            data["acc_y"] *= ascale
            data["acc_z"] *= ascale

            f_xyz.writelines("{0:d}\n".format(self.total_num_ptcls))
            f_xyz.writelines("name x y z vx vy vz ax ay az\n")
            savetxt(
                f_xyz,
                data[["names", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "acc_x", "acc_y", "acc_z"]],
                fmt="%s %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e",
            )

        f_xyz.close()

    def dump_potfit_config(
        self, phase: str = "production", dump_start: int = 0, dump_end: int = None, dump_skip: int = 1
    ):
        """Write configuration files for PotFit by reading the npz dumps.

        Parameters
        ----------
        phase : str
            Phase from which to read dumps. 'equilibration' or 'production'.

        dump_start : int
            Step number from which to start saving. Default is 0.

        dump_end: int
            Last step number to save. Default is None.

        dump_skip : int
            Interval of dumps to skip. Default is 1

        """

        if phase == "equilibration":
            pot_fit_dir = join(self.equilibration_dir, "potfit_configs")
            dump_dir = self.eq_dump_dir
            dump_step = self.eq_dump_step

        elif phase == "magnetization":
            pot_fit_dir = join(self.magnetization_dir, "potfit_configs")
            dump_dir = self.mag_dump_dir
            dump_step = self.mag_dump_step

        else:
            pot_fit_dir = join(self.production_dir, "potfit_configs")
            dump_dir = self.prod_dump_dir
            dump_step = self.prod_dump_step

        if not exists(pot_fit_dir):
            mkdir(pot_fit_dir)

        # Get particles info
        params = self.read_pickle_single("parameters")
        masses = zeros(params.total_num_ptcls)
        sp_start = 0
        sp_end = 0
        for sp_num, sp_m in zip(params.species_num, params.species_masses):
            sp_end += sp_num
            masses[sp_start:sp_end] = sp_m
            sp_start += sp_num

        # Read the list of dumps and sort them in the correct (natural) order
        dumps = listdir(dump_dir)
        dumps.sort(key=num_sort)
        dumps_dict = {}
        for i in dumps:
            s = i.strip("checkpoint_")
            key = s.strip(".npz")
            dumps_dict[int(key)] = i

        if not dump_end:
            dump_end = len(dumps) * dump_step

        dump_skip *= dump_step

        for i in trange(dump_start, dump_end, dump_skip, disable=not self.verbose):
            dump = dumps_dict[i]
            file_name = join(dump_dir, dump)
            data = np_load(file_name, allow_pickle=True)

            # TODO: Find a more pythonic way of doing this
            data["acc"][:,0] *= masses
            data["acc"][:,1] *= masses
            data["acc"][:,2] *= masses

            fname = join(pot_fit_dir, f"config_{i}.out")
            f_xyz = open(fname, "w")
            f_xyz.writelines(f"#N {params.total_num_ptcls:d} 1\n")
            f_xyz.writelines(f"#C {' '.join(n for n in params.species_names)} \n")
            f_xyz.writelines(f"#X {params.Lx} 0.0 0.0\n")
            f_xyz.writelines(f"#Y 0.0 {params.Ly} 0.0\n")
            f_xyz.writelines(f"#Z 0.0 0.0 {params.Lz}\n")
            f_xyz.writelines(f"#E 0.0 \n")
            f_xyz.writelines(f"#F\n")

            savetxt(
                f_xyz,
                c_[data["id"], data["pos"], data["acc"]],
                fmt="%i %.6e %.6e %.6e %.6e %.6e %.6e",
            )

            f_xyz.close()

    def datetime_stamp(self):
        """Add a Date and Time stamp to the log file."""

        if exists(self.log_file):
            with open(self.log_file, "a+") as f_log:
                # Add some space to better distinguish the new beginning
                print(f"\n\n\n", file=f_log)

        with open(self.log_file, "a+") as f_log:
            ct = datetime.datetime.now()
            print(f"{'':~^80}", file=f_log)
            print(f"Date: {ct.year} - {ct.month} - {ct.day}", file=f_log)
            print(f"Time: {ct.hour}:{ct.minute}:{ct.second}", file=f_log)
            print(f"{'':~^80}", file=f_log)

    def file_header(self):
        """Create the log file and print the figlet if not a restart run."""

        self.datetime_stamp()
        if not self.restart:
            with open(self.log_file, "a+") as f_log:
                figlet_obj = Figlet(font="starwars")
                print(figlet_obj.renderText("Sarkas"), file=f_log)
                print("An open-source pure-Python molecular dynamics suite for non-ideal plasmas.", file=f_log)

        # Print figlet to screen if verbose
        if self.verbose:
            self.screen_figlet()

    def from_dict(self, input_dict: dict):
        """
        Update attributes from input dictionary.

        Parameters
        ----------
        input_dict: dict
            Dictionary to be copied.

        """
        self.__dict__.update(input_dict)

    def from_yaml(self, filename: str):
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

                if key == "electrostatic_equilibration":
                    self.electrostatic_equilibration = value

        # rdf_nbins can be defined in either Parameters or Postprocessing. However, Postprocessing will always
        # supersede Parameters choice.
        if "Observables" in dics.keys():
            for i in dics["Observables"]:
                if "RadialDistributionFunction" in i.keys():
                    dics["Parameters"]["rdf_nbins"] = i["RadialDistributionFunction"]["no_bins"]

        return dics

    def make_directory_tree(self):
        """Construct the tree of directory paths of the MD simulation and save it into :attr:`directory_tree`. Set :attr:`job_dir` and :attr:`job_id` too."""

        if self.job_dir is None:
            self.job_dir = basename(self.input_file).split(".")[0]

        if self.job_id is None:
            self.job_id = self.job_dir

        self.job_dir = join(self.md_simulations_dir, self.job_dir)

        # DEV NOTE: The following construction of the dic tree can be put in a loop, but it would be hard to understand hence the hard coding.
        self.directory_tree = {
            "preprocessing": {
                "path": join(self.job_dir, "PreProcessing"),
                "name": "PreProcessing",
                "equilibration": {
                    "path": join(join(self.job_dir, "PreProcessing"), "Equilibration"),
                    "name": "Equilibration",
                    "dumps": {"path": join(join(join(self.job_dir, "PreProcessing"), "Equilibration"), "dumps")},
                },
                "production": {
                    "path": join(join(self.job_dir, "PreProcessing"), "Production"),
                    "name": "Production",
                    "dumps": {"path": join(join(join(self.job_dir, "PreProcessing"), "Production"), "dumps")},
                },
                "magnetization": {
                    "path": join(join(self.job_dir, "PreProcessing"), "Magnetization"),
                    "name": "Magnetization",
                    "dumps": {"path": join(join(join(self.job_dir, "PreProcessing"), "Magnetization"), "dumps")},
                },
            },
            "simulation": {
                "path": join(self.job_dir, "Simulation"),
                "name": "Simulation",
                "equilibration": {
                    "path": join(join(self.job_dir, "Simulation"), "Equilibration"),
                    "name": "Equilibration",
                    "dumps": {"path": join(join(join(self.job_dir, "Simulation"), "Equilibration"), "dumps")},
                },
                "production": {
                    "path": join(join(self.job_dir, "Simulation"), "Production"),
                    "name": "Production",
                    "dumps": {"path": join(join(join(self.job_dir, "Simulation"), "Production"), "dumps")},
                },
                "magnetization": {
                    "path": join(join(self.job_dir, "Simulation"), "Magnetization"),
                    "name": "Magnetization",
                    "dumps": {"path": join(join(join(self.job_dir, "Simulation"), "Magnetization"), "dumps")},
                },
            },
            "postprocessing": {
                "path": join(self.job_dir, "PostProcessing"),
                "name": "PostProcessing",
                "equilibration": {
                    "path": join(join(self.job_dir, "Simulation"), "Equilibration"),
                    "name": "Equilibration",
                    "dumps": {"path": join(join(join(self.job_dir, "Simulation"), "Equilibration"), "dumps")},
                },
                "production": {
                    "path": join(join(self.job_dir, "Simulation"), "Production"),
                    "name": "Production",
                    "dumps": {"path": join(join(join(self.job_dir, "Simulation"), "Production"), "dumps")},
                },
                "magnetization": {
                    "path": join(join(self.job_dir, "Simulation"), "Magnetization"),
                    "name": "Magnetization",
                    "dumps": {"path": join(join(join(self.job_dir, "Simulation"), "Magnetization"), "dumps")},
                },
            },
        }

    def make_directories(self):
        """Make directories in :attr:`directory_tree` where to store the MD results."""

        # Check if the directories exist
        if not exists(self.md_simulations_dir):
            mkdir(self.md_simulations_dir)

        if not exists(self.job_dir):
            mkdir(self.job_dir)

        # Create Process' directories and their subdir
        if self.magnetized and self.electrostatic_equilibration:
            phases = list(self.phase_names_dict.keys())[:3]
        else:
            phases = list(self.phase_names_dict.keys())[:2]

        for _, proc_dict in self.directory_tree.items():
            for key, val in proc_dict.items():
                if key == "path":
                    if not exists(val):
                        mkdir(val)

                elif key in phases:
                    if not exists(val["path"]):
                        mkdir(val["path"])

                    if not exists(val["dumps"]["path"]):
                        mkdir(val["dumps"]["path"])

    def make_files_tree(self):
        """Make the dictionary of files' names and paths, :attr:`filenames_tree`."""
        # Log File
        if self.log_file is None:
            self.filenames_tree["log"]["name"] = f"log_{self.job_id}.out"
        else:
            self.filenames_tree["log"]["name"] = self.log_file

        log_dir = self.directory_tree[self.process]["path"]
        self.filenames_tree["log"]["path"] = join(log_dir, self.filenames_tree["log"]["name"])
        # Redundancy for backward compatibility
        self.log_file = self.filenames_tree["log"]["path"]

        self.filenames_tree["particles"] = {
            "equilibration": {
                "name": "checkpoint_",
                "path": join(self.directory_tree[self.process]["equilibration"]["dumps"]["path"], "checkpoint_"),
            },
            "production": {
                "name": "checkpoint_",
                "path": join(self.directory_tree[self.process]["production"]["dumps"]["path"], "checkpoint_"),
            },
            "magnetization": {
                "name": "checkpoint_",
                "path": join(self.directory_tree[self.process]["magnetization"]["dumps"]["path"], "checkpoint_"),
            },
        }
        self.filenames_tree["thermodynamics"] = {
            "equilibration": {
                "name": f"EquilibrationEnergy_{self.job_id}.csv_",
                "path": join(
                    self.directory_tree[self.process]["equilibration"]["path"], f"EquilibrationEnergy_{self.job_id}.csv"
                ),
            },
            "production": {
                "name": f"ProductionEnergy_{self.job_id}.csv_",
                "path": join(
                    self.directory_tree[self.process]["production"]["path"], f"ProductionEnergy_{self.job_id}.csv"
                ),
            },
            "magnetization": {
                "name": f"MagnetizationEnergy_{self.job_id}.csv_",
                "path": join(
                    self.directory_tree[self.process]["magnetization"]["path"], f"MagnetizationEnergy_{self.job_id}.csv"
                ),
            },
        }

        # Redundancy for faster lookup
        for ph in self.phase_names_dict.keys():
            self.particles_filepaths[ph] = self.filenames_tree["particles"][ph]["path"]
            self.thermodynamics_filepaths[ph] = self.filenames_tree["thermodynamics"][ph]["path"]

        # The following is for old version compatibility
        # Equilibration directory and sub_dir
        self.equilibration_dir = self.directory_tree[self.process]["equilibration"]["path"]
        self.eq_dump_dir = self.directory_tree[self.process]["equilibration"]["dumps"]["path"]
        # self.particles_directories["equilibration"] = self.eq_dump_dir

        # Production dir and sub_dir
        self.production_dir = self.directory_tree[self.process]["production"]["path"]
        self.prod_dump_dir = self.directory_tree[self.process]["production"]["dumps"]["path"]
        # self.particles_directories["production"] = self.prod_dump_dir

        # Production phase filenames
        self.prod_energy_filename = self.filenames_tree["thermodynamics"]["production"]["path"]
        self.prod_ptcls_filename = self.filenames_tree["particles"]["production"]["path"]

        # Equilibration phase filenames
        self.eq_energy_filename = self.filenames_tree["thermodynamics"]["equilibration"]["path"]
        self.eq_ptcls_filename = self.filenames_tree["particles"]["equilibration"]["path"]

        # Magnetic dir
        if self.magnetized and self.electrostatic_equilibration:
            self.magnetization_dir = self.directory_tree[self.process]["magnetization"]["path"]
            self.mag_dump_dir = self.directory_tree[self.process]["magnetization"]["dumps"]["path"]

            # Magnetization phase filenames
            self.mag_energy_filename = self.filenames_tree["thermodynamics"]["magnetization"]["path"]
            self.mag_ptcls_filename = self.filenames_tree["particles"]["magnetization"]["path"]

    def postprocess_info(self, simulation, observable=None):
        pass
        """
        Print Post-processing info to file in a reader-friendly format.

        Parameters
        ----------
        simulation : :class:`sarkas.processes.PostProcess`
            PostProcess class.

        observable : str
            Observable whose info to print. Default = None.
            Choices = ['header','rdf', 'ccf', 'dsf', 'ssf', 'vm']

        """
        # choices = ["header", "rdf", "ccf", "dsf", "ssf", "vd", "vacf", "p_tensor", "ec", "diff_flux"]
        # msg = (
        #     "Observable not defined.\n"
        #     "Please choose an observable from this list\n"
        #     "'rdf' = Radial Distribution Function,\n"
        #     "'ccf' = Current Correlation Function,\n"
        #     "'dsf' = Dynamic Structure Function,\n"
        #     "'ssf' = Static Structure Factor,\n"
        #     "'vd' = Velocity Distribution\n",
        #     "'vacf' = Velocity Auto Correlation Function\n"
        #     "'ec' = Electric Current\n"
        #     "'diff_flux' = Diffusion Flux\n"
        #     "'p_tensor' = Pressure Tensor",
        # )
        # if observable is None:
        #     raise ValueError(msg)
        #
        # if observable not in choices:
        #     raise ValueError(msg)
        #
        # screen = sys.stdout
        # f_log = open(self.filenames_tree["log"]["path"], "a+")
        # # redirect printing to file
        # sys.stdout = f_log
        #
        # if observable == "header":
        #     # Header of process
        #     process_title = f"{self.process.capitalize():^80}"
        #     print(f"{'':*^80}\n {process_title} \n{'':*^80}")

        # elif observable == "rdf":
        #     msg = simulation.rdf.pretty_print_msg()
        # elif observable == "ssf":
        #     msg = simulation.ssf.pretty_print_msg()
        # elif observable == "dsf":
        #     msg = simulation.dsf.pretty_print_msg()
        # elif observable == "ccf":
        #     msg = simulation.ccf.pretty_print_msg()
        # elif observable == "vacf":
        #     msg = simulation.vacf.pretty_print_msg()
        # elif observable == "p_tensor":
        #     msg = simulation.p_tensor.pretty_print_msg()
        # elif observable == "ec":
        #     msg = simulation.ec.pretty_print_msg()
        # elif observable == "diff_flux":
        #     msg = simulation.diff_flux.pretty_print_msg()
        # elif observable == "vd":
        #     simulation.vm.setup(simulation.parameters)
        #     msg = (
        #         f"\nVelocity Moments:\n"
        #         f"Maximum no. of moments = {simulation.vm.max_no_moment}\n"
        #         f"Maximum velocity moment = {int(2 * simulation.vm.max_no_moment)}"
        #     )
        #
        # print(msg)
        # sys.stdout = screen
        # f_log.close()

    @staticmethod
    def potential_info(simulation):
        """
        Print potential information.

        Parameters
        ----------
        simulation : :class:`sarkas.processes.Process`
            Process class containing the potential info and other parameters.

        """
        warn(
            "Deprecated feature. It will be removed in a future release. Use potential.pot_pretty_print()",
            category=DeprecationWarning,
        )
        simulation.potential.pot_pretty_print(simulation.potential)

    def directory_size_report(self, sizes, process):
        """Print the estimated file sizes."""

        screen = sys.stdout
        f_log = open(self.filenames_tree["log"]["path"], "a+")
        repeat = 2 if self.verbose else 1

        # redirect printing to file
        sys.stdout = f_log
        msg_h = " Filesize Estimates "

        while repeat > 0:
            msg = f"\n\n{msg_h:=^70}\n"

            if self.equilibration_phase:
                size_GB, size_MB, size_KB, rem = convert_bytes(sizes[0, 0])
                msg += (
                    f"\nEquilibration:\n"
                    f"\tCheckpoint filesize: {int(size_GB)} GB {int(size_MB)} MB {int(size_KB)} KB {int(rem)} bytes\n"
                )

                size_GB, size_MB, size_KB, rem = convert_bytes(sizes[0, 1])
                msg += f"\tCheckpoint folder size: {int(size_GB)} GB {int(size_MB)} MB {int(size_KB)} KB {int(rem)} bytes"

            if self.magnetized and self.electrostatic_equilibration:
                size_GB, size_MB, size_KB, rem = convert_bytes(sizes[2, 0])
                msg += (
                    f"\nMagnetization:\n"
                    f"\tCheckpoint filesize: {int(size_GB)} GB {int(size_MB)} MB {int(size_KB)} KB {int(rem)} bytes\n"
                )

                size_GB, size_MB, size_KB, rem = convert_bytes(sizes[2, 1])
                msg += f"\tCheckpoint folder size: {int(size_GB)} GB {int(size_MB)} MB {int(size_KB)} KB {int(rem)} bytes"

            size_GB, size_MB, size_KB, rem = convert_bytes(sizes[1, 0])
            msg += (
                f"\nProduction:\n"
                f"\tCheckpoint filesize: {int(size_GB)} GB {int(size_MB)} MB {int(size_KB)} KB {int(rem)} bytes\n"
            )

            size_GB, size_MB, size_KB, rem = convert_bytes(sizes[1, 1])
            msg += f"\tCheckpoint folder size: {int(size_GB)} GB {int(size_MB)} MB {int(size_KB)} KB {int(rem)} bytes\n"

            size_GB, size_MB, size_KB, rem = convert_bytes(sizes[:, 1].sum())
            if process == "preprocessing":
                msg += f"\nTotal minimum required space: {int(size_GB)} GB {int(size_MB)} MB {int(size_KB)} KB {int(rem)} bytes"
            else:
                msg += f"\nTotal occupied space: {int(size_GB)} GB {int(size_MB)} MB {int(size_KB)} KB {int(rem)} bytes"

            print(msg)

            repeat -= 1
            sys.stdout = screen

        f_log.close()

    def preprocess_timing(self, str_id, t, loops):
        """Print times estimates of simulation to file first and then to screen if verbose."""
        screen = sys.stdout
        f_log = open(self.filenames_tree["log"]["path"], "a+")
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

            elif str_id in ["PP", "PM", "FMM"]:
                print(f"Time of {str_id} acceleration calculation averaged over {loops - 1} steps:")
                print(f"{int(t_min)} min {int(t_sec)} sec {int(t_msec)} msec {int(t_usec)} usec {int(t_nsec)} nsec \n")

            elif str_id in ["Equilibration", "Magnetization", "Production"]:
                print(f"Time of a single {str_id} step averaged over {loops - 1} steps:")
                print(f"{int(t_min)} min {int(t_sec)} sec {int(t_msec)} msec {int(t_usec)} usec {int(t_nsec)} nsec \n")
                if str_id == "Production":
                    print("\n\n{:-^70} \n".format(" Total Estimated Times "))
            repeat -= 1
            sys.stdout = screen

        f_log.close()

    @staticmethod
    def read_npz(self, fldr: str, filename: str):
        """
        Load particles' data from dumps.

        Parameters
        ----------
        fldr : str
            Folder containing dumps.

        filename: str
            Name of the dump file to load.

        Returns
        -------
        struct_array : numpy.ndarray
            Structured data array.

        Raises
        ------
            : DeprecationWarning
        """

        warn(
            "Deprecated feature. It will be removed in a future release.\nUse :meth:`read_particles_npz`.",
            category=DeprecationWarning,
        )
        return InputOutput.read_particles_npz(fldr, filename)

    @staticmethod
    def read_particles_npz(fldr: str, filename: str):
        """
        Load particles' data from dumps.

        Parameters
        ----------
        fldr : str
            Folder containing dumps.

        filename: str
            Name of the dump file to load.

        Returns
        -------
        struct_array : numpy.ndarray
            Structured data array.

        """

        file_name = join(fldr, filename)
        data = np_load(file_name, allow_pickle=True)
        # Dev Notes: the old way of saving the xyz file by
        # savetxt(f_xyz, np.c_[data["names"],data["pos"] ....]
        # , fmt="%10s %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e")
        # was not working, because the columns of np.c_[] all have the data type <U32
        # which is in conflict with the desired fmt. i.e. data["names"] was not recognized as a string.
        # So I have to create a new structured array and pass this. I could not think of a more Pythonic way.
        struct_array = zeros(
            data["names"].size,
            dtype=[
                ("names", "U6"),
                ("id", int),
                ("pos_x", float64),
                ("pos_y", float64),
                ("pos_z", float64),
                ("vel_x", float64),
                ("vel_y", float64),
                ("vel_z", float64),
                ("acc_x", float64),
                ("acc_y", float64),
                ("acc_z", float64),
            ],
        )
        struct_array["names"] = data["names"]
        struct_array["id"] = data["id"]
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

    def read_pickle(self, process, dir_path: str = None):
        """
        Read pickle files containing all the simulation information.

        Parameters
        ----------
        process : :class:`sarkas.processes.Process`
            Process class containing MD run info to save.

        dir_path : str, (optional)
            Path to directory where pickles are stored.

        """

        file_list = ["parameters", "integrator", "potential"]

        if dir_path:
            directory = dir_path
        else:
            directory = self.directory_tree[self.process]["path"]

        for fl in file_list:
            filename = join(directory, fl + ".pickle")
            with open(filename, "rb") as handle:
                data = pickle.load(handle)
                process.__dict__[fl] = copy(data)

        # Read species
        filename = join(directory, "species.pickle")
        process.species = []
        with open(filename, "rb") as handle:
            data = pickle.load(handle)
            process.species = copy(data)

    def read_pickle_single(self, class_to_read: str, dir_path: str = None):
        """
        Read the desired pickle file.

        Parameters
        ----------
        class_to_read : str
            Name of the class to read.

        dir_path : str, (optional)
            Path to directory where pickles are stored.

        Returns
        -------
        _copy : cls
            Copy of desired class.

        """
        # Redirect to the correct process folder

        if dir_path:
            directory = dir_path
        else:
            directory = self.directory_tree[self.process]["path"]

        filename = join(directory, class_to_read + ".pickle")
        with open(filename, "rb") as pickle_file:
            data = pickle.load(pickle_file)
            _copy = deepcopy(data)
        return _copy

    def save_pickle(self, simulation):
        """
        Save all simulations parameters in pickle files.

        Parameters
        ----------
        simulation : :class:`sarkas.processes.Process`
            Process class containing MD run info to save.
        """
        file_list = ["parameters", "integrator", "potential", "species"]

        # Redirect to the correct process folder
        if self.process == "preprocessing":
            indx = 0
        else:
            # Note that Postprocessing needs the link to simulation's folder
            # because that is where I look for energy files and pickle files
            indx = 1

        for fl in file_list:
            filename = join(self.directory_tree[self.process]["path"], fl + ".pickle")
            with open(filename, "wb") as pickle_file:
                pickle.dump(simulation.__dict__[fl], pickle_file)
                pickle_file.close()

    @staticmethod
    def screen_figlet():
        """
        Print a colored figlet of Sarkas to screen.
        """
        if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
            # Assume white background in Jupyter Notebook
            clr = DARK_COLORS[randint(0, len(DARK_COLORS))]
        else:
            # Assume dark background in IPython/Python Kernel
            clr = LIGHT_COLORS[randint(0, len(LIGHT_COLORS))]
        fnt = FONTS[randint(0, len(FONTS))]
        print_figlet("\nSarkas\n", font=fnt, colors=clr)

        print("\nAn open-source pure-python molecular dynamics suite for non-ideal plasmas.\n\n")

    def setup(self):
        """Create file paths and directories for the simulation."""
        self.make_directory_tree()
        self.make_directories()
        self.make_files_tree()
        self.file_header()

    def setup_checkpoint(self, params):
        """
        Assign attributes needed for saving dumps.

        Parameters
        ----------
        params : :class:`sarkas.core.Parameters`
            General simulation parameters.

        species : :class:`sarkas.plasma.Species`
            List of Species classes.

        """

        self.copy_params(params)
        dkeys = ["Time", "Total Energy", "Total Kinetic Energy", "Total Potential Energy", "Temperature"]

        if len(self.thermodynamics_to_save) > 0:
            extra_cols = []
            if "Pressure" in self.thermodynamics_to_save:
                extra_cols.append("Total Pressure")
                extra_cols.append("Ideal Pressure")
                extra_cols.append("Excess Pressure")
            if "Enthalpy" in self.thermodynamics_to_save:
                extra_cols.append("Total Enthalpy")

            for col in extra_cols:
                dkeys.append(col)

        # Create the Energy file
        if len(self.species_names) > 1:
            for i, sp_name in enumerate(self.species_names):
                dkeys.append("{} Kinetic Energy".format(sp_name))
                dkeys.append("{} Potential Energy".format(sp_name))
                dkeys.append("{} Temperature".format(sp_name))
                if len(self.thermodynamics_to_save) > 0:
                    if "Pressure" in self.thermodynamics_to_save:
                        dkeys.append("{} Total Pressure".format(sp_name))
                        dkeys.append("{} Ideal Pressure".format(sp_name))
                        dkeys.append("{} Excess Pressure".format(sp_name))
                    if "Enthalpy" in self.thermodynamics_to_save:
                        dkeys.append("{} Enthalpy".format(sp_name))

        data = dict.fromkeys(dkeys)
        # Check whether energy files exist already
        if not exists(self.prod_energy_filename):
            with open(self.prod_energy_filename, "w+") as f:
                w = csv.writer(f)
                w.writerow(data.keys())

        if not exists(self.eq_energy_filename) and not params.load_method[-7:] == "restart":
            with open(self.eq_energy_filename, "w+") as f:
                w = csv.writer(f)
                w.writerow(data.keys())

        if self.magnetized and self.electrostatic_equilibration:
            if not exists(self.mag_energy_filename) and not params.load_method[-7:] == "restart":
                data = dict.fromkeys(dkeys)
            with open(self.mag_energy_filename, "w+") as f:
                w = csv.writer(f)
                w.writerow(data.keys())

    def simulation_summary(self, simulation):
        """
        Print out to file a summary of simulation's parameters.
        If verbose output then it will print twice: the first time to file and second time to screen.

        Parameters
        ----------
        simulation : :class:`sarkas.processes.Process`
            Simulation's parameters

        """

        screen = sys.stdout
        f_log = open(self.filenames_tree["log"]["path"], "a+")

        repeat = 2 if self.verbose else 1
        # redirect printing to file
        sys.stdout = f_log

        # Print to file first then to screen if repeat == 2
        while repeat > 0:
            # Header of process
            process_title = f"{self.process.capitalize():^80}"
            print(f"\n\n{'':*^80}")
            print(process_title)
            print(f"{'':*^80}")

            print(f"\nJob ID: {self.job_id}")
            print(f"Job directory: {self.job_dir}")
            print(
                f"{self.directory_tree[self.process]['name']} directory: \n{self.directory_tree[self.process]['path']}"
            )

            print(
                f"\nEquilibration dumps directory: \n{self.directory_tree[self.process]['equilibration']['dumps']['path']}"
            )
            print(f"Production dumps directory: \n{self.directory_tree[self.process]['production']['dumps']['path']}")

            print(
                f"\nEquilibration Thermodynamics file: \n{self.filenames_tree['thermodynamics']['equilibration']['path']}"
            )
            print(f"Production Thermodynamics file: \n{self.filenames_tree['thermodynamics']['production']['path']}")

            if simulation.parameters.load_method in ["production_restart", "prod_restart"]:
                print(f"\n\n{' Production Restart ':~^80}")
                ct = datetime.datetime.now()
                print(f"Date: {ct.year} - {ct.month} - {ct.day}")
                print(f"Time: {ct.hour}:{ct.minute}:{ct.second}")
                # Simulation Phases
                simulation.parameters.pretty_print()
                # Integrator
                simulation.integrator.pretty_print()

            elif simulation.parameters.load_method in ["equilibration_restart", "eq_restart"]:
                print(f"\n\n{' Equilibration Restart ':~^80}")
                ct = datetime.datetime.now()
                print(f"Date: {ct.year} - {ct.month} - {ct.day}")
                print(f"Time: {ct.hour}:{ct.minute}:{ct.second}")
                # Simulation Phases
                simulation.parameters.pretty_print()
                # Integrator
                simulation.integrator.pretty_print()

            elif simulation.parameters.load_method in ["magnetization_restart", "mag_restart"]:
                print(f"\n\n{' Magnetization Restart ':~^80}")
                ct = datetime.datetime.now()
                print(f"Date: {ct.year} - {ct.month} - {ct.day}")
                print(f"Time: {ct.hour}:{ct.minute}:{ct.second}")
                # Simulation Phases
                simulation.parameters.pretty_print()
                # Integrator
                simulation.integrator.pretty_print()

            elif self.process in ["simulation", "preprocessing"]:
    
                print("\nPARTICLES:")
                print(f"Total No. of particles = {simulation.parameters.total_num_ptcls}")

                print(f"No. of species = {len(simulation.parameters.species_num)}")
                # The line below is to prevent printing electron background in the case of LJ potential
                species_to_print = simulation.species[:-1] if simulation.potential.type == "lj" else simulation.species
                for isp, sp in enumerate(species_to_print):
                    if sp.name != "electron_background":
                        print("Species ID: {}".format(isp))
                    sp.pretty_print(simulation.potential.type, simulation.parameters.units)

                # Parameters Info
                simulation.parameters.pretty_print()
                # Potential Info
                simulation.potential.pretty_print()
                # Integrator
                simulation.integrator.pretty_print()

            repeat -= 1
            sys.stdout = screen  # Restore the original sys.stdout

        f_log.close()

    @staticmethod
    def time_info(simulation):
        """
        Print time simulation's parameters.

        Parameters
        ----------
        simulation : :class:`sarkas.processes.Process`
            Process class containing the timing info and other parameters.

        """
        warn(
            "Deprecated feature. It will be removed in a future release.\n" "Use Integrator.pretty_print()",
            category=DeprecationWarning,
        )

        simulation.integrator.pretty_print()

    def time_stamp(self, time_stamp: str, t: tuple):
        """
        Print out to screen elapsed times. If verbose output, print to file first and then to screen.

        Parameters
        ----------
        time_stamp : str
            Array of time stamps.

        t : tuple
            Time in hrs, min, sec, msec, usec, nsec..
        """
        screen = sys.stdout
        f_log = open(self.filenames_tree["log"]["path"], "a+")
        repeat = 2 if self.verbose else 1
        t_hrs, t_min, t_sec, t_msec, t_usec, t_nsec = t
        # redirect printing to file
        sys.stdout = f_log

        while repeat > 0:
            if "Particles Initialization" in time_stamp:
                print("\n\n{:-^70} \n".format(" Initialization Times "))
            # elif "Production" in time_stamp:
            #     print("\n\n{:-^70} \n".format(" Phases Times "))
            if t_hrs == 0 and t_min == 0 and t_sec <= 2:
                print(f"\n{time_stamp} Time: {int(t_sec)} sec {int(t_msec)} msec {int(t_usec)} usec {int(t_nsec)} nsec")
            else:
                print(f"\n{time_stamp} Time: {int(t_hrs)} hrs {int(t_min)} min {int(t_sec)} sec")

            repeat -= 1
            sys.stdout = screen

        f_log.close()

    def timing_study(self, simulation):
        """
        Info specific for timing study.

        Parameters
        ----------
        simulation : :class:`sarkas.processes.Process`
            Process class containing the info to print.

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

    def write_to_logger(self, message):
        """Append the message to log file."""
        with open(self.log_file, "a+") as f_log:
            # redirect printing to file
            print(message, file=f_log)


def alpha_to_int(text):
    """Convert strings of numbers into integers.

    Parameters
    ----------
    text : str
        Text to be converted into an int, if `text` is a number.

    Returns
    -------
    _ : int, str
        Integral number otherwise returns a string.

    """
    return int(text) if text.isdigit() else text


def num_sort(text):
    """
    Sort strings with numbers inside.

    Parameters
    ----------
    text : str
        Text to be split into str and int

    Returns
    -------
     : list
        List containing text and integers

    Notes
    -----
    Function copied from
    https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside.
    Originally from http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)

    """

    return [alpha_to_int(c) for c in re.split(r"(\d+)", text)]


def convert_bytes(tot_bytes):
    """Convert bytes to human-readable GB, MB, KB.

    Parameters
    ----------
    tot_bytes : int
        Total number of bytes.

    Returns
    -------
    [GB, MB, KB, rem] : list
        Bytes divided into Giga, Mega, Kilo bytes.
    """
    GB, rem = divmod(tot_bytes, 1024 * 1024 * 1024)
    MB, rem = divmod(rem, 1024 * 1024)
    KB, rem = divmod(rem, 1024)

    return [GB, MB, KB, rem]


def print_to_logger(message, log_file, print_to_screen: bool = False):
    """Print observable useful info to log file and to screen if `self.verbose` is `True`.

    Parameters
    ----------
    message : str
        Message to append to log and screen.

    log_file: str
        Path to log file.

    print_to_screen : bool
        Flag for printing to screen. Default = `False`.

    """

    screen = sys.stdout
    f_log = open(log_file, "a+")
    repeat = 2 if print_to_screen else 1

    # redirect printing to file
    sys.stdout = f_log
    while repeat > 0:
        print(message)
        repeat -= 1
        sys.stdout = screen

    f_log.close()
