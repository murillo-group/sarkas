from sarkas.base import Simulation
from sarkas.tools.testing import PreProcess
input_file_name = 'sarkas/examples/yukawa_icf_new.yaml'

args = {"input_file": input_file_name}

sim = Simulation()
sim.common_parser(input_file_name)
other_input = {'Integrator': {'type': 'Verlet'} }
sim.setup(other_inputs=other_input)
sim.run()
# vrb = Verbose(params)
# vrb.sim_setting_summary(params)
#simulation.run(params)
