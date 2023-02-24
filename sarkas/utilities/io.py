"""
Module handling the I/O for an MD run.
"""
import csv
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
from os import listdir, mkdir
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

    electrostatic_equilibration: bool = False
    eq_dump_dir: str = "dumps"
    equilibration_dir: str = "Equilibration"
    input_file: str = None  # MD run input file.
    job_dir: str = None
    job_id: str = None
    log_file: str = None
    mag_dump_dir: str = "dumps"
    magnetization_dir: str = "Magnetization"
    magnetized: bool = False
    preprocess_file: str = None
    preprocessing: bool = False
    preprocessing_dir: str = "PreProcessing"
    process: str = "preprocessing"
    processes_dir: str = None
    prod_dump_dir: str = "dumps"
    production_dir: str = "Production"
    postprocessing_dir: str = "PostProcessing"
    simulations_dir: str = "Simulations"
    simulation_dir: str = "Simulation"
    verbose: bool = False
    xyz_dir: str = None
    xyz_filename: str = None

    def __init__(self, process: str = "preprocess"):
        self.process = process

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
            "Deprecated feature. It will be removed in the v2.0.0 release. Use potential.method_pretty_print()",
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
        """Create all directories', subdirectories', and files' paths."""

        if self.job_dir is None:
            self.job_dir = basename(self.input_file).split(".")[0]

        if self.job_id is None:
            self.job_id = self.job_dir

        self.job_dir = join(self.simulations_dir, self.job_dir)

        # Create Processes directories
        self.processes_dir = [
            join(self.job_dir, self.preprocessing_dir),
            join(self.job_dir, self.simulation_dir),
            join(self.job_dir, self.postprocessing_dir),
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
        self.equilibration_dir = join(self.processes_dir[indx], self.equilibration_dir)
        self.eq_dump_dir = join(self.equilibration_dir, "dumps")
        # Production dir and sub_dir
        self.production_dir = join(self.processes_dir[indx], self.production_dir)
        self.prod_dump_dir = join(self.production_dir, "dumps")

        # Production phase filenames
        self.prod_energy_filename = join(self.production_dir, "ProductionEnergy_" + self.job_id + ".csv")
        self.prod_ptcls_filename = join(self.prod_dump_dir, "checkpoint_")

        # Equilibration phase filenames
        self.eq_energy_filename = join(self.equilibration_dir, "EquilibrationEnergy_" + self.job_id + ".csv")
        self.eq_ptcls_filename = join(self.eq_dump_dir, "checkpoint_")

        # Magnetic dir
        if self.electrostatic_equilibration:
            self.magnetization_dir = join(self.processes_dir[indx], self.magnetization_dir)
            self.mag_dump_dir = join(self.magnetization_dir, "dumps")
            # Magnetization phase filenames
            self.mag_energy_filename = join(self.magnetization_dir, "MagnetizationEnergy_" + self.job_id + ".csv")
            self.mag_ptcls_filename = join(self.mag_dump_dir, "checkpoint_")

        if self.process == "postprocessing":
            indx = 2  # Redirect to the correct folder

        # Log File
        if self.log_file is None:
            self.log_file = join(self.processes_dir[indx], "log_" + self.job_id + ".out")
        else:
            self.log_file = join(self.processes_dir[indx], self.log_file)

    def dump(self, phase, ptcls, it):
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
        if phase == "production":
            ptcls_file = self.prod_ptcls_filename + str(it)
            tme = it * self.dt
            savez(
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
            savez(
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
            savez(
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
            "Total Energy": kinetic_energies.sum() + ptcls.potential_energy,
            "Total Kinetic Energy": kinetic_energies.sum(),
            "Potential Energy": ptcls.potential_energy,
            "Total Temperature": ptcls.species_num.transpose() @ temperatures / ptcls.total_num_ptcls,
        }
        if len(temperatures) > 1:
            for sp, kin in enumerate(kinetic_energies):
                data[f"{self.species_names[sp]} Kinetic Energy"] = kin
                data[f"{self.species_names[sp]} Potential Energy"] = potential_energies[sp]
                data[f"{self.species_names[sp]} Temperature"] = temperatures[sp]

        with open(energy_file, "a") as f:
            w = csv.writer(f)
            w.writerow(data.values())

    def dump_xyz(self, phase: str = "production", dump_start: int = 0, dump_end: int = None, dump_skip: int = 1) -> None:
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

        # Rescale constants. This is needed since OVITO has a small number limit.
        pscale = 1.0 / self.a_ws
        vscale = 1.0 / (self.a_ws * self.total_plasma_frequency)
        ascale = 1.0 / (self.a_ws * self.total_plasma_frequency**2)

        f_xyz = open(self.xyz_filename, "w+")

        # Read the list of dumps and sort them in the correct (natural) order
        dumps = listdir(dump_dir)
        dumps.sort(key=num_sort)
        dumps_dict = {}
        for i in dumps:
            _, key = i.strip("_")
            dumps_dict[int(key)] = i

        if not dump_end:
            dump_end = len(dumps) * dump_step

        dump_skip *= dump_step

        for i in trange(dump_start, dump_end, dump_skip, disable=not self.verbose):
            dump = dumps_dict[i]
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
            savetxt(f_xyz, data, fmt="%s %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e")

        f_xyz.close()

    def dump_potfit_config(
        self, phase: str = "production", dump_start: int = 0, dump_end: int = None, dump_skip: int = 1
    ) -> None:
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
            data = self.read_npz(dump_dir, dump)

            data["acc_x"] *= masses
            data["acc_y"] *= masses
            data["acc_z"] *= masses

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
                c_[data["id"], data["pos_x"], data["pos_y"], data["pos_z"], data["acc_x"], data["acc_y"], data["acc_z"]],
                fmt="%i %.6e %.6e %.6e %.6e %.6e %.6e",
            )

            f_xyz.close()

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

    def make_directories(self):
        """Create directories where to store MD results."""

        # Check if the directories exist
        if not exists(self.simulations_dir):
            mkdir(self.simulations_dir)

        if not exists(self.job_dir):
            mkdir(self.job_dir)

        # Create Process' directories and their subdir
        for i in self.processes_dir:
            if not exists(i):
                mkdir(i)
        # The following automatically create directories in the correct Process
        if not exists(self.equilibration_dir):
            mkdir(self.equilibration_dir)

        if not exists(self.eq_dump_dir):
            mkdir(self.eq_dump_dir)

        if not exists(self.production_dir):
            mkdir(self.production_dir)

        if not exists(self.prod_dump_dir):
            mkdir(self.prod_dump_dir)

        if self.electrostatic_equilibration:
            if not exists(self.magnetization_dir):
                mkdir(self.magnetization_dir)

            if not exists(self.mag_dump_dir):
                mkdir(self.mag_dump_dir)

        if self.preprocessing:
            if not exists(self.preprocessing_dir):
                mkdir(self.preprocessing_dir)

        if not exists(self.postprocessing_dir):
            mkdir(self.postprocessing_dir)

    def postprocess_info(self, simulation, write_to_file=False, observable=None):
        """
        Print Post-processing info to file and/or screen in a reader-friendly format.

        Parameters
        ----------
        simulation : :class:`sarkas.processes.PostProcess`
            PostProcess class.

        write_to_file : bool
            Flag for printing info also to file. Default= False.

        observable : str
            Observable whose info to print. Default = None.
            Choices = ['header','rdf', 'ccf', 'dsf', 'ssf', 'vm']

        """

        choices = ["header", "rdf", "ccf", "dsf", "ssf", "vd"]
        msg = (
            "Observable not defined.\n"
            "Please choose an observable from this list\n"
            "'rdf' = Radial Distribution Function,\n"
            "'ccf' = Current Correlation Function,\n"
            "'dsf' = Dynamic Structure Function,\n"
            "'ssf' = Static Structure Factor,\n"
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
                process_title = f"{self.process.capitalize():^80}"
                print(f"{'':*^80}\n {process_title} \n{'':*^80}")

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
                msg = (
                    f"\nVelocity Moments:\n"
                    f"Maximum no. of moments = {simulation.vm.max_no_moment}\n"
                    f"Maximum velocity moment = {int(2 * simulation.vm.max_no_moment)}"
                )
                print(msg)

            repeat -= 1

            if write_to_file:
                sys.stdout = screen

        if write_to_file:
            f_log.close()

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

    def preprocess_sizing(self, sizes):
        """Print the estimated file sizes."""

        screen = sys.stdout
        f_log = open(self.log_file, "a+")
        repeat = 2 if self.verbose else 1

        # redirect printing to file
        sys.stdout = f_log
        while repeat > 0:
            msg_h = " Filesize Estimates "
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
            msg += (
                f"\nTotal minimum required space: {int(size_GB)} GB {int(size_MB)} MB {int(size_KB)} KB {int(rem)} bytes"
            )

            print(msg)

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
    def read_npz(fldr: str, filename: str):
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
        # was not working, because the columns of np.c_[] all have the same data type <U32
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

    def read_pickle(self, process):
        """
        Read pickle files containing all the simulation information.

        Parameters
        ----------
        process : :class:`sarkas.processes.Process`
            Process class containing MD run info to save.

        """
        file_list = ["parameters", "integrator", "potential"]

        # Redirect to the correct process folder
        if self.process == "preprocessing":
            indx = 0
        else:
            # Note that Postprocessing needs the link to simulation's folder
            # because that is where I look for energy files and pickle files
            indx = 1

        for fl in file_list:
            filename = join(self.processes_dir[indx], fl + ".pickle")
            with open(filename, "rb") as handle:
                data = pickle.load(handle)
                process.__dict__[fl] = copy(data)

        # Read species
        filename = join(self.processes_dir[indx], "species.pickle")
        process.species = []
        with open(filename, "rb") as handle:
            data = pickle.load(handle)
            process.species = copy(data)

    def read_pickle_single(self, class_to_read: str):
        """
        Read the desired pickle file.

        Parameters
        ----------
        class_to_read : str
            Name of the class to read.

        Returns
        -------
        _copy : cls
            Copy of desired class.

        """
        # Redirect to the correct process folder
        if self.process == "preprocessing":
            indx = 0
        else:
            # Note that Postprocessing needs the link to simulation's folder
            # because that is where I look for energy files and pickle files
            indx = 1

        filename = join(self.processes_dir[indx], class_to_read + ".pickle")
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
            filename = join(self.processes_dir[indx], fl + ".pickle")
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
        self.create_file_paths()
        self.make_directories()
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
        # Check whether energy files exist already
        if not exists(self.prod_energy_filename):
            # Create the Energy file
            dkeys = ["Time", "Total Energy", "Total Kinetic Energy", "Potential Energy", "Temperature"]
            if len(self.species_names) > 1:
                for i, sp_name in enumerate(self.species_names):
                    dkeys.append("{} Kinetic Energy".format(sp_name))
                    dkeys.append("{} Potential Energy".format(sp_name))
                    dkeys.append("{} Temperature".format(sp_name))
            data = dict.fromkeys(dkeys)

            with open(self.prod_energy_filename, "w+") as f:
                w = csv.writer(f)
                w.writerow(data.keys())

        if not exists(self.eq_energy_filename) and not params.load_method[-7:] == "restart":
            # Create the Energy file
            dkeys = ["Time", "Total Energy", "Total Kinetic Energy", "Potential Energy", "Temperature"]
            if len(self.species_names) > 1:
                for i, sp_name in enumerate(self.species_names):
                    dkeys.append("{} Kinetic Energy".format(sp_name))
                    dkeys.append("{} Potential Energy".format(sp_name))
                    dkeys.append("{} Temperature".format(sp_name))
            data = dict.fromkeys(dkeys)

            with open(self.eq_energy_filename, "w+") as f:
                w = csv.writer(f)
                w.writerow(data.keys())

        if self.electrostatic_equilibration:
            if not exists(self.mag_energy_filename) and not params.load_method[-7:] == "restart":
                # Create the Energy file
                dkeys = ["Time", "Total Energy", "Total Kinetic Energy", "Potential Energy", "Temperature"]
                if len(self.species_names) > 1:
                    for i, sp_name in enumerate(self.species_names):
                        dkeys.append("{} Kinetic Energy".format(sp_name))
                        dkeys.append("{} Potential Energy".format(sp_name))
                        dkeys.append("{} Temperature".format(sp_name))
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

                print(f"\nJob ID: {self.job_id}")
                print(f"Job directory: {self.job_dir}")
                print(f"PostProcessing directory: \n{self.postprocessing_dir}")

                print(f"\nEquilibration dumps directory: {self.eq_dump_dir}")
                print(f"Production dumps directory: \n{self.prod_dump_dir}")

                print(f"\nEquilibration Thermodynamics file: \n{self.eq_energy_filename}")
                print(f"Production Thermodynamics file: \n{self.prod_energy_filename}")

            else:

                # Header of process
                process_title = "{:^80}".format(self.process.capitalize())
                print("\n\n")
                print(*["*" for i in range(50)])
                print(process_title)
                print(*["*" for i in range(50)])

                print(f"\nJob ID: {self.job_id}")
                print(f"Job directory: {self.job_dir}")
                print(f"\nEquilibration dumps directory: \n", {self.eq_dump_dir})
                print(f"Production dumps directory: \n", {self.prod_dump_dir})

                print(f"\nEquilibration Thermodynamics file: \n{self.eq_energy_filename}")
                print(f"Production Thermodynamics file: \n{self.prod_energy_filename}")

                print("\nPARTICLES:")
                print(f"Total No. of particles = {simulation.parameters.total_num_ptcls}")

                print(f"No. of species = {len(simulation.parameters.species_num)}")
                # This line below is to prevent printing electron background in the case of LJ potential
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

        screen = sys.stdout
        f_log = open(self.log_file, "a+")
        repeat = 2 if self.verbose else 1
        # redirect printing to file
        sys.stdout = f_log

        while repeat > 0:
            print(message)

            repeat -= 1
            sys.stdout = screen

        f_log.close()


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
