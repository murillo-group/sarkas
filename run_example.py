from sarkas.processes import PreProcess, Simulation, PostProcess
#from sarkas.tools.transport import TransportCoefficient
# from numpy.random import Generator, PCG64
#
input_file_name = 'sarkas/examples/'
#
# rg = Generator(PCG64(12345))
#
# # i = 0
# # args = {
# #     "IO":
# #         {
# #             "job_id": "ocp_mks_run{}".format(i),
# #             "job_dir": "ocp_mks_run{}".format(i)
# #         },
# # }
# #
# preproc = PreProcess(input_file_name)
# preproc.setup(read_yaml=True)
# preproc.run(loops=5)
# #
# for i in range(10):
#     args = {
#         "IO":
#             {
#                 "simulation_dir": 'ICF_example',
#                 "job_id": "icf_mks_run{}".format(i),
#                 "job_dir": "icf_mks_run{}".format(i)
#             },
#         "Parameters":
#             {"rand_seed": rg.integers(0, 1598765198)}
#     }
# #
#     sim = Simulation(input_file_name)
#     sim.setup(read_yaml=True, other_inputs=args)
#     sim.run()
# #     #
# #
# for i in range(10):
#     args = {
#         "IO":
#             {
#                 "simulation_dir": 'ICF_example',
#                 "job_id": "icf_mks_run{}".format(i),
#                 "job_dir": "icf_mks_run{}".format(i)
#             },
#         "Parameters":
#             {"rand_seed": rg.integers(0, 1598765198)}
#     }
# #
postproc = PostProcess(input_file_name)
postproc.setup(read_yaml=True)
#
#     postproc.therm.setup(postproc.parameters)
#     postproc.therm.temp_energy_plot(postproc, phase='equilibration', show=False)
#     postproc.therm.temp_energy_plot(postproc, phase='production', show=False)
#     postproc.therm.plot('Temperature', show=False)
#     # Radial Distribution Function
#     postproc.rdf.setup(postproc.parameters)
#     postproc.rdf.save()
#     postproc.rdf.plot(show=False)
#     # Hermite Coeff
#     postproc.hc.setup(postproc.parameters, 'equilibration')
#     postproc.hc.parse()
#     postproc.hc.plot(show=False)
#     #
#     postproc.hc.setup(postproc.parameters, 'production')
#     postproc.hc.parse()
#     postproc.hc.plot(show=False)
#     # Velocit moments
#     postproc.vm.setup(postproc.parameters, 'equilibration')
#     postproc.vm.parse()
#     postproc.vm.plot_ratios(show=False)
#     #
#     postproc.vm.setup(postproc.parameters, 'production')
#     postproc.vm.parse()
#     postproc.vm.plot_ratios(show=False)
#
#     # postproc.ssf.setup(postproc.parameters)
#     # postproc.ssf.compute()
#     # postproc.ssf.plot(show=True)
# Dynamic Structure Function
postproc.dsf.setup(postproc.parameters)
postproc.dsf.compute()
#     # postproc.dsf.plot(show=False, dispersion=True)
#     # #
#     # postproc.ccf.setup(postproc.parameters)
#     # postproc.ccf.parse()
#     # postproc.ccf.plot(show=False)
#
#     diffusion = TransportCoefficient.diffusion(postproc.parameters,
#                                                phase='production',
#                                                show=False)
#     interdiffusion = TransportCoefficient.interdiffusion(postproc.parameters,
#                                                phase='production',
#                                                show=False)