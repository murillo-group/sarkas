#!/usr/bin/env python
# coding: utf-8

import os 
from numpy.random import Generator, Philox

from sarkas.processes import PostProcess, Simulation
from sarkas.tools.observables import RadialDistributionFunction, Thermodynamics
import re

def find_largest_checkpoint(directory):
    # Regex pattern to match files in the format 'checkpoint_i.npz'
    pattern = r'checkpoint_(\d+)\.npz'

    max_checkpoint = -1
    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        if match:
            checkpoint_number = int(match.group(1))
            if checkpoint_number > max_checkpoint:
                max_checkpoint = checkpoint_number

    if max_checkpoint == -1:
        return None
    else:
        print( f'checkpoint_{max_checkpoint}.npz' in os.listdir(directory))
        return max_checkpoint

    
def sim_restart(input_file_name:str, job_name:str, phase: str, restart_step : int = None, max_steps: int = None): 
    """Restart a simulation that ended prematurely.
    
    Parameters:
    -----------
    input_file_name : str
        The name of the input file.
    job_name : str  
        Job name used as job directory name.
    phase : str 
        The phase of the simulation to restart.
    restart_step : int
        The step at which to restart the simulation.
    max_steps : int
        The maximum number of steps to run the simulation.
    
    Returns:
        None
    """
    # Get info and parameters from simulation
    
    args = {
        "IO":
            {
                "verbose": False,
                "job_dir" : job_name,
            },
    }
    postproc = PostProcess(input_file_name)
    postproc.setup(read_yaml = True,  other_inputs = args)


    if phase == "production":
        if restart_step: 
            prod_steps = max_steps if max_steps else postproc.parameters.production_steps
            args = {
                "Parameters": {
                    "equilibration_phase": False,
                    "load_method": "production_restart",
                    "restart_step": restart_step,
                    "production_steps": prod_steps
                    },
                "IO":
                    {
                        "verbose": False,
                        "job_dir": job_name,
                    },
            }
        sim = Simulation(input_file_name)
        sim.setup(read_yaml = True, other_inputs = args)
        sim.run()

    args = {
#         "Parameters" : {"rand_seed": seed,
#                             "equilibration_steps": n_eq},
        "IO":
            {
                "verbose": False,
                "job_dir" : job_name,
            },
    }
    postproc = PostProcess(input_file_name)
    postproc.setup(read_yaml = True,  other_inputs = args)

    
    directory = postproc.io.directory_tree["simulation"][phase]["dumps"]["path"]
    largest_checkpoint = find_largest_checkpoint(directory)
    
    if largest_checkpoint < max_steps:
        args = {
            "Parameters": {
                "equilibration_phase": False,
                "load_method": "production_restart",
                "restart_step": largest_checkpoint,
                "production_steps": max_steps
                },
            "IO":
                {
                    "verbose": False,
                    "job_dir": job_name,
                },
        }
        sim = Simulation(input_file_name)
        sim.setup(read_yaml = True, other_inputs = args)
        sim.run()

def postprocessing(job_name, input_file_name):
    """
    Perform post-processing tasks on the simulation results.

    Args:
        job_name (int): Job name used as job directory name.
        input_file_name (str): The name of the input file.

    Returns:
        None
    """

    # Define the arguments for post-processing
    args = {
        "IO": {
            "verbose": False,
            "job_dir": job_name # Job name used as job directory name, if not provided, the job_dir will be the input file name.
        },
    }

    # Create a PostProcess object
    postproc = PostProcess(input_file_name)
    postproc.setup(read_yaml=True, other_inputs=args)

    therm = Thermodynamics()
    therm.setup(postproc.parameters)
    therm.compute() 
    # Make the Temperature-Energy plots
    therm.temp_energy_plot(postproc)
    
    # Radial Distribution Functions
    rdf = RadialDistributionFunction()
    rdf.setup(postproc.parameters)
    rdf.compute()

    # Make a plot of the RDF
    ax = rdf.plot(
        scaling=rdf.a_ws,  # Scale by the Wigner-Seitz radius
        y = [("C-C RDF", "Mean")], # Column name
        xlabel = r'$r/a_{\rm ws}$', 
        ylabel = r'$g(r)$'
    )
    ax.legend(["C-C RDF"])  # Remake the legend.
    # The plot will be saved automatically.

if __name__ == "__main__":

    input_file = os.path.join("input_files","yocp_quickstart.yaml")
    
    # Spawn ten random number generators
    rng_s = Generator(Philox())

    # Read this link for more information on why we use Philox
    # https://numpy.org/doc/stable/reference/random/parallel.html

    # Generate a random seed and number of equilibration steps using the random number generator
    seed = rng_s.integers(987650,341865651)
    
    job_name = "yocp_quickstart"
    
    sim_restart(job_name, input_file)
    
    postprocessing(job_name, input_file)
