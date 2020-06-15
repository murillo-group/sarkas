"""
Module for plotting observables.
"""
import sys
from S_params import Params
import S_postprocessing as Observable

input_file = sys.argv[1]
params = Params()
params.setup(input_file)

# params.PostProcessing.no_ka_values = 15
# E = Observable.Thermodynamics(params)
# E.plot('Total Energy', delta=True, show=False)
# E.plot('Temperature', False)
# E.plot('Gamma', False)
#E.plot('Pressure Tensor ACF', False, True)

# K = E.dataframe["Kinetic Energy"]  # pandas df


# params.PostProcessing.ssf_no_ka_values = [15, 15, 15]
# SSF = Observable.StaticStructureFactor(params)
# SSF.compute()
# SSF.plot(show=True)

# params.PostProcessing.dsf_no_ka_values = [5, 5, 5]
DSF = Observable.DynamicStructureFactor(params)
# DSF.compute()
DSF.plot(show=False, dispersion=False)

# #
#rdf = Observable.RadialDistributionFunction(params)
# rdf.parse()
#rdf.plot()

# lw = 2
# fsz = 14
# r, g = np.loadtxt('OCP_mks/RDF_OCP_mks_old.out', unpack = True)
# fig, ax = plt.subplots(1,1, figsize = (10, 7) )
# ax.plot(rdf.dataframe["ra values"], rdf.dataframe["H-H RDF"], lw = 2, label = 'New')
# ax.plot(r, g, lw = 2, label = 'Old')
# ax.grid(True, alpha=0.3)
# ax.legend(loc='best', fontsize=fsz)
#
# ax.tick_params(labelsize=fsz)
# ax.set_ylabel(r'$g(r)$', fontsize=fsz)
# ax.set_xlabel(r'$r/a$', fontsize=fsz)
# # ax.set_ylim(0, 5)
# fig.tight_layout()
# fig.show()

# J = Observable.ElectricCurrent(params)
# J.dataframe["Total Electrical Current"]
# J.plot()

# sigma = Observable.TransportCoefficients(params)
# sigma.compute("Electrical Conductivity")
# P = Observable.PressureTensor(params)
# P.plot(True)

# XYZ = Observable.XYZFile(params)
# XYZ.save()
