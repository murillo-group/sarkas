from sarkas.base import Simulation
from sarkas.tools.preprocessing import PreProcess
from sarkas.tools.postprocessing import PostProcess
input_file_name = 'sarkas/examples/binary_unp.yaml'

args = {"input_file": input_file_name}

# sim = ()
sim = Simulation(input_file_name)
# other_input = {'Integrator': {'type': 'Verlet'} }
sim.setup(read_yaml=True)
# sim.hc.setup(sim.parameters, sim.species)
# sim.hc.compute()
sim.run()
# sim.initialization()
# vrb = Verbose(params)
# vrb.sim_setting_summary(params)
#simulation.run(params)
