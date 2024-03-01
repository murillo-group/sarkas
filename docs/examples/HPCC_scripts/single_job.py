#!/usr/bin/env python
# coding: utf-8

import os 
from numpy.random import Generator, Philox

from sarkas.processes import PostProcess, Simulation
from sarkas.tools.observables import RadialDistributionFunction, Thermodynamics


def simulate(seed: int, job_name: str,  input_file_name: str):
    """
    Run an MD simulation.

    Parameters:
        seed (int): The random seed for particle initialization.
        job_name (int): Job name used as job directory name.
        input_file_name (str): The name of the input file.

    Returns:
        None
    """
    args = {
        "Parameters": {
            "rand_seed": seed,          # Random seed for particle initialization
        },
        "IO": {
            "verbose": False,   # No need to output the progress bar.
            "job_dir": job_name, # Job name used as job directory name, if not provided, the job_dir will be the input file name.
            # Note that we don't want to overwrite each simulation
            # So we save each simulation in its own folder by passing
            # a dictionary of dictionary with folder's name
        },
    }
    sim = Simulation(input_file_name)
    sim.setup(read_yaml=True, other_inputs=args)
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
    
    simulate(seed, job_name, input_file)
    
    postprocessing(job_name, input_file)
