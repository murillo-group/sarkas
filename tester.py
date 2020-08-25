from sarkas.base import Simulation
from sarkas.tools.testing import PreProcess
input_file_name = 'sarkas/examples/binary_unp.yaml'

args = {"input_file": input_file_name}

# sim = ()
sim = Simulation()
sim.common_parser(input_file_name)
# other_input = {'Integrator': {'type': 'Verlet'} }
sim.setup()
sim.run()
# sim.initialization()
# vrb = Verbose(params)
# vrb.sim_setting_summary(params)
#simulation.run(params)
