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
E = Observable.Thermodynamics(params)
# E.plot('Total Energy', True)
# E.plot('Temperature', False)
# E.plot('Gamma', False)
# E.plot('Pressure Tensor ACF', False, True)
#
# K = E.dataframe["Kinetic Energy"]  # pandas df

DSF = Observable.DynamicStructureFactor(params)
DSF.plot(True)
# #
# rdf = Observable.RadialDistributionFunction(params)
# rdf.plot()

# J = Observable.ElectricCurrent(params)
# #J.dataframe["Total Electrical Current"]
# J.plot()

# sigma = Observable.TransportCoefficients(params)
# sigma.compute("Electrical Conductivity")
# P = Observable.PressureTensor(params)
# P.plot(True)

# XYZ = Observable.XYZFile(params)
# XYZ.save()
