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
params = Observable.read_pickle(input_file)

E = Observable.Thermodynamics(params)
E.parse()
E.boxplot('Total Energy', show=False)
#
# Z = Observable.VelocityAutocorrelationFunctions(params)
# Z.compute()
# Z.plot(intercurrent=True, show=True)
#
# params.PostProcessing.dsf_no_ka_values = [5, 5, 5]
# DSF = Observable.DynamicStructureFactor(params)
# DSF.compute()
# DSF.plot(show=False)
# #
# SSF = Observable.StaticStructureFactor(params)
# SSF.compute()
# SSF.plot(show=True)
#
# rdf = Observable.RadialDistributionFunction(params)
# data = Observable.load_from_restart(params.Control.dump_dir, params.Control.Nsteps)
# rdf.save(data["rdf_hist"])
# rdf.plot(show=True)
# rdf.parse()
# rdf.plot()
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