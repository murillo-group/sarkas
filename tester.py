from sarkas.base import Simulation
from sarkas.io.base import Verbose

input_file_name = 'sarkas/examples/egs_cgs_new.yaml'

args = {"input_file": input_file_name}

sim = Simulation()
sim.common_parser(input_file_name)
sim.setup()
sim.run()
# vrb = Verbose(params)
# vrb.sim_setting_summary(params)
#simulation.run(params)
