from sarkas.processes import PreProcess, Simulation, PostProcess

input_file_name = 'sarkas/examples/coulomb_bim_mks_new.yaml'
#
# for i in range(5):
#
#     args = {"IO": {"job_id": "run_{}".format(i),
#                    "job_dir": "run_{}".format(i)}
#             }
#
# preproc = PreProcess(input_file_name)
# preproc.setup(read_yaml=True)
# preproc.run(loops=5)

sim = Simulation(input_file_name)
sim.setup(read_yaml=True)
sim.run()

#
postproc = PostProcess(input_file_name)
postproc.setup(read_yaml=True)
# postproc.rdf.setup(postproc.parameters, postproc.species)
# postproc.rdf.save()
# postproc.rdf.plot(show=True)
# postproc.hc.setup(postproc.parameters, postproc.species)
# postproc.hc.compute()
# postproc.hc.plot(show=True)
postproc.parameters.thermostat_type = postproc.thermostat.type
postproc.parameters.thermostat_tau = postproc.thermostat.tau
postproc.therm.setup(postproc.parameters, postproc.species)
postproc.therm.temp_energy_plot(postproc, phase='equilibration', show=True)
postproc.therm.temp_energy_plot(postproc, phase='production', show=True)
postproc.therm.plot('Temperature', show=True)