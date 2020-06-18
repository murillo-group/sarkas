"""
Module for plotting observables.
"""
import sys
from S_params import Params
import S_postprocessing as Observable

lw = 2
fsz = 14

input_file = sys.argv[1]

params = Params()
params.common_parser(input_file)

# SSF = Observable.StaticStructureFactor(params)
# SSF.compute()
# SSF.plot(show=True)

# rdf = Observable.RadialDistributionFunction(params)
# rdf.parse()
# rdf.plot()
#
# J = Observable.ElectricCurrent(params)
# J.dataframe["Total Electrical Current"]
# J.plot()
#
# sigma = Observable.TransportCoefficients(params)
# sigma.compute("Electrical Conductivity")
#
# XYZ = Observable.XYZFile(params)
# XYZ.save()
