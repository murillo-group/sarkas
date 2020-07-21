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
op.add_option("-t", "--pre_run_testing", action='store_true', dest='test', default=False, help="Test input parameters")
op.add_option("-v", "--verbose", action='store_true', dest='verbose', default=False, help="Verbose output")
op.add_option("-p", "--plot_show", action='store_true', dest='plot_show', default=False, help="Show plots")
op.add_option("-c", "--check", type='choice', choices=['therm', 'prod'],
              action='store', dest='check', help="Check current state of run")
op.add_option("-d", "--job_dir", action='store', dest='job_dir', help="Job Directory")
op.add_option("-j", "--job_id", action='store', dest='job_id', help="Job ID")
op.add_option("-s", "--seed", action='store', dest='seed', type='int', help="Random Number Seed")
op.add_option("-i", "--input", action='store', dest='input_file', help="YAML Input file")
op.add_option("-r", "--restart", action='store', dest='restart', type=int, help="Restart simulation")

options, _ = op.parse_args()

# Input file is a must
if options.input_file is None:
    raise OSError('Input file not defined.')

# Read initial conditions and setup parameters
params = Params()
params.setup(options)

if options.check is not None:
    params = PostProc.read_pickle(params.Control.checkpoint_dir)
else:
    # Update rand seed with option. This supersedes the input file.
    if options.seed:
        params.load_rand_seed = int(options.seed)

    # Verbose output. This does not supersede the input file if False.
    # That is if you don't give this option and the input file has Control.verbose=Yes, then you will
    # still have a verbose output
    if options.verbose:
        params.Control.verbose = True
        
     if options.restart is not None:                                                                                            
        params.load_method = 'restart'                                                                                           
        params.load_restart_step = options.restart                                                                               
                                                   
# Test/Check/Run
if options.test:
    Testing.main(params, options.repeat)
elif options.check is not None:
    if options.check == "therm":
        T = PostProc.Thermalization(params)
        T.temp_energy_plot(params, options.plot_show)
        # hc = T.hermite_plot(params, options.plot_show)
        # vm = T.moment_ratios_plot(params, options.plot_show)
    elif options.check == "prod":
        E = PostProc.Thermodynamics(params)
        E.temp_energy_plot(params, options.plot_show)
else:
    Simulation.main(params)
