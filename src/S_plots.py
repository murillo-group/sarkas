"""
Module for plotting observables.
"""
import sys
from S_params import Params
import S_postprocessing as Observable

input_file = sys.argv[1]

params = Params()
params.setup(input_file)

params.PostProcessing.no_ka_values = 15
E = Observable.Thermodynamics(params)
E.plot('Total Energy', True)
E.plot('Temperature', False)
E.plot('Gamma', False)
# E.plot('Pressure', False)

# K = E.dataframe["Kinetic Energy"]

SSF = Observable.StaticStructureFactor(params)
SSF.compute(False)
SSF.plot()
# #
# rdf = Observable.RadialDistributionFunction(params)
# rdf.plot()

# J = Observable.ElectricCurrent(params)
# #J.dataframe["Total Electrical Current"]
# J.plot()

# P = Observable.PressureTensor(params)
# P.plot(True)

# XYZ = Observable.XYZFile(params)
# XYZ.save()
