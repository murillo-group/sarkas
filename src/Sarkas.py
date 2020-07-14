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
from optparse import OptionParser
# Sarkas modules
import S_testing as Testing
import S_simulation as Simulation
import S_postprocessing as PostProc
from S_params import Params

# Construct the argument parser
op = OptionParser()

# Add the arguments to the parser
op.add_option("-t", "--pre_run_testing", action='store_true', dest='test', default=False, help="Pre Run Testing Flag")
op.add_option("-c", "--check", type='choice', choices=['therm', 'prod'],
              action='store', dest='check', help="Check current state of run")
op.add_option("-d", "--job_dir", action='store', dest='job_dir', help="Job Directory")
op.add_option("-j", "--job_id", action='store', dest='job_id', help="Job ID")
op.add_option("-s", "--seed", action='store', dest='seed', type='int', help="Random Number Seed")
op.add_option("-v", "--verbose", action='store_true', dest='verbose', help="Verbose output")
op.add_option("-i", "--input", action='store', dest='input_file', help="YAML Input file")

options, arguments = op.parse_args()

if options.input_file is None:
    raise OSError('Input file not defined.')

# Read initial conditions and setup parameters
params = Params()
params.setup(options)

# Update rand seed with option
if options.seed:
    params.load_rand_seed = int(options.seed)

# Test/Run/Check
if options.test:
    Testing.main(params)
elif options.check is not None:
    if options.check == "therm":
        T = PostProc.Thermalization(params)
        T.temp_energy_plot(show=True)
        hc = T.hermite_plot(params, True)
        vm = T.moment_ratios_plot(params, True)
    elif options.check == "prod":
        E = PostProc.Thermodynamics(params)
        E.temp_energy_plot(show=True)
else:
    Simulation.main(params)
