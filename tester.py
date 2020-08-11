from sarkas.simulation.params import Params
from sarkas.simulation import simulation
from sarkas.io.verbose import Verbose

input_file_name = 'sarkas/examples/yocp_tau01.yaml'

args = {"input_file": input_file_name}

params = Params()
params.setup(args)

vrb = Verbose(params)
vrb.sim_setting_summary(params)
#simulation.run(params)
