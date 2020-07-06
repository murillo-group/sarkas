"""
Module for plotting observables.
"""
import sys
import matplotlib.pyplot as plt
import os
import S_postprocessing as Observable

plt.style.use(
    os.path.join(os.path.join(os.getcwd(), 'src'), 'PUBstyle'))

input_file = sys.argv[1]
#
# params = Params()
# params.setup(input_file)  # Read initial conditions and setup parameters
# save_file = open(os.path.join(params.Control.checkpoint_dir, "S_parameters.pickle"), "wb")
# pickle.dump(params, save_file)
# save_file.close()
params = Observable.read_pickle(input_file)
# params.Control.checkpoint_dir = os.path.join("Simulations", params.Control.checkpoint_dir)
E = Observable.Thermodynamics(params)
E.parse()
# E.boxplot('Total Energy', show=True)
# # E.statistics("Total Energy", max_no_divisions=1000, show=True)
# # E.boxplot("Total Energy", True)
#
# # clr = [24/255, 69/255, 49/255]
# # clr2 = [240/255, 133/255, 33/255]
#
#
# # E.statistics("Temperature", max_no_divisions=1000, show=True)
# # params.Control.Nsteps = 10000
# Z = Observable.VelocityAutocorrelationFunctions(params)
# # Z.compute()
# Z.plot(intercurrent=True, show=True)


# params.PostProcessing.dsf_no_ka_values = [5, 5, 5]
# DSF = Observable.DynamicStructureFactor(params)
# DSF.compute()
# DSF.plot(show=False)
# #
# SSF = Observable.StaticStructureFactor(params)
# SSF.compute()
# SSF.plot(show=True)

# rdf = Observable.RadialDistributionFunction(params)
# data = Observable.load_from_restart(params.Control.dump_dir, params.Control.Nsteps)
# rdf.save(data["rdf_hist"])
# rdf.plot(show=True)
# rdf.parse()
# rdf.plot()
#
# J = Observable.ElectricCurrent(params)
# J.dataframe["Total Electrical Current"]
# J.plot()

# TC = Observable.TransportCoefficients(params)
# # TC.compute("Electrical Conductivity",show=True)
# TC.compute("Diffusion",show=True)
# #
# XYZ = Observable.XYZWriter(params)
# XYZ.save()