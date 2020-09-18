from sarkas.processes import PreProcess, Simulation, PostProcess
from numpy.random import Generator, PCG64

input_file_name = 'sarkas/examples/yukawa_icf.yaml'

rg = Generator(PCG64(12345))

i = 0
args = {
    "IO":
        {
            "job_id": "icf_run{}".format(i),
            "job_dir": "icf_run{}".format(i)
        },
}

preproc = PreProcess(input_file_name)
preproc.setup(read_yaml=True, other_inputs=args)
preproc.run(loops=5)

for i in range(3):
    args = {
        "IO":
            {
                "job_id": "icf_run{}".format(i),
                "job_dir": "icf_run{}".format(i)
            },
        "Parameters":
            {"rand_seed": rg.integers(0, 1598765198)}
    }

    sim = Simulation(input_file_name)
    sim.setup(read_yaml=True, other_inputs=args)
    sim.run()
    #

for i in range(3):
    args = {
        "IO":
            {
                "job_id": "icf_run{}".format(i),
                "job_dir": "icf_run{}".format(i)
            },
        "Parameters":
            {"rand_seed": rg.integers(0, 1598765198)}
    }

    postproc = PostProcess(input_file_name)
    postproc.setup(read_yaml=True, other_inputs=args)
    #
    postproc.therm.setup(postproc.parameters, postproc.species)
    postproc.therm.temp_energy_plot(postproc, phase='equilibration', show=True)
    postproc.therm.temp_energy_plot(postproc, phase='production', show=True)
    postproc.therm.plot('Temperature', show=True)
    #
    postproc.rdf.setup(postproc.parameters)
    postproc.rdf.save()
    postproc.rdf.plot(show=True)
    #
    postproc.hc.setup(postproc.parameters, 'equilibration')
    postproc.hc.parse()
    postproc.hc.plot(show=True)
    #
    postproc.hc.setup(postproc.parameters, 'production')
    postproc.hc.parse()
    postproc.hc.plot(show=True)
    #
    postproc.vm.setup(postproc.parameters, 'equilibration')
    postproc.vm.parse()
    postproc.vm.plot_ratios(show=True)
    #
    postproc.vm.setup(postproc.parameters, 'production')
    postproc.vm.parse()
    postproc.vm.plot_ratios(show=True)
    #
    postproc.ssf.setup(postproc.parameters)
    postproc.ssf.parse()
    postproc.ssf.plot(show=True)
    #
    postproc.dsf.setup(postproc.parameters)
    postproc.dsf.parse()
    postproc.dsf.plot(show=True)
    #
    postproc.ccf.setup(postproc.parameters)
    postproc.ccf.parse()
    postproc.ccf.plot(show=True)
