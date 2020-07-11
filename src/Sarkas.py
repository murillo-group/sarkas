"""
                                SARKAS: 1.0

An open-source pure-python molecular dynamics (MD) code for simulating plasmas.

Developed by the research group of:
Professor Michael S. Murillo
murillom@msu.edu
Dept. of Computational Mathematics, Science, and Engineering,
Michigan State University
"""

# Python modules
import argparse
# Sarkas modules
import S_testing as Testing
import S_simulation as Simulation
import S_postprocessing as PostProc
from S_params import Params

# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-i", "--input", required=True, help="YAML Input file")
ap.add_argument("-c", "--check", required=False, help="Check current state of run")
ap.add_argument("-t", "--pre_run_testing", required=False, help="Pre Run Testing Flag")
ap.add_argument("-d", "--job_dir", required=False, help="Job Directory")
ap.add_argument("-j", "--job_id", required=False, help="Job ID")
ap.add_argument("-s", "--seed", required=False, help="Random Number Seed")
args = vars(ap.parse_args())

pre_run_testing = False if args["pre_run_testing"] is None else True

# Read initial conditions and setup parameters
params = Params()
params.setup(args)

# Update rand seed with option
if not args["seed"] is None:
    params.load_rand_seed = int(args["seed"])

# Test/Run/Check
if pre_run_testing:
    Testing.main(params)

elif not args["check"] is None:
    if args["check"] == 'therm':
        T = PostProc.Thermalization(params)
        T.temp_energy_plot(show=True)
    else:
        E = PostProc.Thermodynamics(params)
        E.temp_energy_plot(show=True)
else:
    Simulation.main(params)
