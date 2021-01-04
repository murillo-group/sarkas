from sarkas.processes import Simulation, PostProcess
from sarkas.processes import PreProcess
from sarkas.tools.transport import TransportCoefficient
# from numpy.random import Generator, PCG64

input_file_name = 'sarkas/examples/coulomb_bim_mks.yaml'
#
# rg = Generator(PCG64(12345))

preproc = PreProcess(input_file_name)
preproc.setup(read_yaml=True)
preproc.run(loops=50)

#
# sim = Simulation(input_file_name)
# sim.setup(read_yaml=True)
# sim.run()

# postproc = PostProcess(input_file_name)
# postproc.setup(read_yaml=True)
#
# postproc.rdf.setup(postproc.parameters)
# postproc.rdf.compute()
# postproc.rdf.plot(show=True)
# # #
# postproc.therm.setup(postproc.parameters)
# postproc.therm.temp_energy_plot(postproc, phase='equilibration', show=True)
# postproc.therm.temp_energy_plot(postproc, phase='production', show=True)

# postproc.hc.setup(postproc.parameters, 'equilibration')
# # postproc.hc.parse()
# postproc.hc.compute()
# postproc.hc.plot(show=True)

# postproc.hc.setup(postproc.parameters, 'production')
# # postproc.hc.parse()
# postproc.hc.compute()
# postproc.hc.plot(show=True)
#
# # postproc.vm.setup(postproc.parameters, 'equilibration')
# # postproc.vm.parse()
# # postproc.vm.compute()
# # postproc.vm.plot_ratios(show=False)
# #
# # postproc.vm.setup(postproc.parameters, 'production')
# # #postproc.vm.parse()
# # postproc.vm.compute()
# # postproc.vm.plot_ratios(show=False)
# #
# postproc.dsf.setup(postproc.parameters)
# postproc.dsf.compute()
# postproc.dsf.plot(show=True)
# # #
# # postproc.ssf.setup(postproc.parameters)
# # postproc.ssf.parse()
# # postproc.ssf.plot(show=False)
# # #
# postproc.ccf.setup(postproc.parameters)
# postproc.ccf.compute()
# # postproc.ccf.plot(show=False)
#
#
# diffusion = TransportCoefficient.diffusion(postproc.parameters,
#                                            phase='production',
#                                            show=True)
# interdiffusion = TransportCoefficient.interdiffusion(postproc.parameters,
#                                            phase='production',
