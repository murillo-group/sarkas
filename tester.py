from sarkas.simulation.params import Params
from sarkas.simulation import simulation
from sarkas.io.verbose import Verbose
import sarkas.tools.postprocessing as PostProc

input_file_name = 'sarkas/examples/EGS_cgs.yaml'

args = {"input_file": input_file_name,
        "job_dir": 'EGS_cgs',
        "job_id": 'EGS_cgs'}

params = Params()
params.setup(args)

simulation.run(params)

E = PostProc.Thermodynamics(params)
E.temp_energy_plot(params,show=True)
E.temp_energy_plot(params, phase='production', show=True)

j = PostProc.ElectricCurrent(params)
j.plot(show=True)

ccf = PostProc.CurrentCorrelationFunctions(params)
ccf.plot(show=True, longitudinal=True)
ccf.plot(show=True, longitudinal=False)

dsf = PostProc.DynamicStructureFactor(params)
dsf.plot(show=True)

ssf = PostProc.StaticStructureFactor(params)
ssf.plot(show=True)

hc = PostProc.HermiteCoefficients(params)
hc.plot(show=True)

vacf = PostProc.VelocityAutocorrelationFunctions(params)
vacf.compute()
vacf.plot(show=True)

vm = PostProc.VelocityMoments(params)
vm.compute()
vm.plot_ratios(show=True)

sigma = PostProc.Transport.compute(params, "Electrical Conductivity", True)
diff = PostProc.Transport.compute(params, "Diffusion", True)
eta = PostProc.Transport.compute(params, "Viscosity", True)
