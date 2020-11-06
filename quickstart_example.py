from sarkas.processes import Simulation, PostProcess
import matplotlib.pyplot as plt
#from sarkas.processes import PreProcess
#from sarkas.tools.transport import TransportCoefficient

input_file_name = 'sarkas/examples/egs_cgs.yaml'

# preproc = PreProcess(input_file_name)
# preproc.setup(read_yaml=True)
# preproc.run(loops=5)

#
egs = Simulation(input_file_name)
egs.setup(read_yaml=True)
egs.run()
#
egs_postproc = PostProcess(input_file_name)
egs_postproc.setup(read_yaml=True)
#
egs_postproc.therm.setup(egs_postproc.parameters)
egs_postproc.therm.temp_energy_plot(egs_postproc, phase='equilibration', show=True)
egs_postproc.therm.temp_energy_plot(egs_postproc, phase='production', show=True)

egs_postproc.rdf.setup(egs_postproc.parameters)
egs_postproc.rdf.save()
egs_postproc.rdf.plot(show=False)


args = {
    "IO":
        {
            "job_dir": "yocp_qkst",
            "job_id": "yocp"
        },
    "Potential":
        {'type': 'Yukawa'}
}

yocp = Simulation(input_file_name)
yocp.setup(read_yaml=True, other_inputs=args)
yocp.run()
#
yocp_postproc = PostProcess(input_file_name)
yocp_postproc.setup(read_yaml=True, other_inputs=args)
#
yocp_postproc.therm.setup(yocp_postproc.parameters)
yocp_postproc.therm.temp_energy_plot(yocp_postproc, phase='equilibration', show=True)
yocp_postproc.therm.temp_energy_plot(yocp_postproc, phase='production', show=True)

yocp_postproc.rdf.setup(yocp_postproc.parameters)
yocp_postproc.rdf.save()
yocp_postproc.rdf.plot(show=False)


import pandas as pd

pre_paper_yocp = pd.read_csv('egs_rdf.csv', index_col=False)
pre_paper_egs = pd.read_csv('yocp_rdf.csv', index_col=False)
# plt.style.use('MSUstyle')
fig, ax = plt.subplots(1,1)
ax.plot(pre_paper_egs['distance']*1e-10, pre_paper_egs['rdf'], '--', label = 'PRE EGS')
ax.plot(egs_postproc.rdf.dataframe['distance']*1e-2, egs_postproc.rdf.dataframe['Al-Al RDF'], label = 'Sarkas EGS')
ax.plot(pre_paper_yocp['distance']*1e-10, pre_paper_yocp['rdf'], '--', label = 'PRE YOCP')
ax.plot(yocp_postproc.rdf.dataframe['distance']*1e-2, yocp_postproc.rdf.dataframe['Al-Al RDF'], label = 'Sarkas YOCP')
ax.legend()
fig.show()

# postproc.therm.plot('Temperature', show=False)
# #
# # postproc.hc.setup(postproc.parameters, 'equilibration')
# # # postproc.hc.parse()
# # postproc.hc.compute()
# # postproc.hc.plot(show=False)
# #
# # postproc.hc.setup(postproc.parameters, 'production')
# # # postproc.hc.parse()
# # postproc.hc.compute()
# # postproc.hc.plot(show=False)
# #
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
# # postproc.dsf.setup(postproc.parameters)
# # postproc.dsf.parse()
# # postproc.dsf.plot(show=False)
# # #
# # postproc.ssf.setup(postproc.parameters)
# # postproc.ssf.parse()
# # postproc.ssf.plot(show=False)
# # #
# # postproc.ccf.setup(postproc.parameters)
# # postproc.ccf.parse()
# # postproc.ccf.plot(show=False)
#
# postproc.vacf.setup(postproc.parameters)
# postproc.vacf.compute()
#
# diffusion = TransportCoefficient.diffusion(postproc.parameters,
#                                            phase='production',
#                                            show=True)