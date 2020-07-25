"""
Module for plotting observables.
"""
import sys
import matplotlib.pyplot as plt
# import os
# Sarkas Module
import sarkas.tools.postprocessing as Observable
# plt.style.use(
#     os.path.join(os.path.join(os.getcwd(), 'src'), 'MSUstyle'))

plt.close('all')
fig, ax = plt.subplots(1, 1)
for i, input_file in enumerate(sys.argv[1:], 1):
    params = Observable.read_pickle(input_file)
    # T.moment_ratios_plot(params, show=True)
    E = Observable.Thermodynamics(params)
    E.parse()
    E.temp_energy_plot(params, show=True)
    # #
    # Z = Observable.VelocityAutocorrelationFunctions(params)
    # Z.compute()
    # Z.plot(intercurrent=True, show=True)
    
    # params.PostProcessing.dsf_no_ka_values = [5, 5, 5]
    # DSF = Observable.DynamicStructureFactor(params)
    # DSF.compute()
    # DSF.plot(show=False)
    # #
    # SSF = Observable.StaticStructureFactor(params)
    # SSF.compute()
    # SSF.plot(show=True)
    #
    rdf = Observable.RadialDistributionFunction(params)
    # data = Observable.load_from_restart(params.Control.dump_dir, params.Control.Nsteps)
    #df.save(data["rdf_hist"])
    #rdf.plot(show=True)
    rdf.parse()
    rdf.plot(show=True)
    #
    # J = Observable.ElectricCurrent(params)
    # J.compute()
    # J.plot()
    # TC = Observable.TransportCoefficients(params)
    # TC.compute("Electrical Conductivity", show=True)
    # TC.compute("Diffusion",show=True)
    # #
    # XYZ = Observable.XYZWriter(params)
    # XYZ.save()
    
