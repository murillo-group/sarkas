"""
Module for plotting observables.
"""
import sys
from S_params import Params
import S_postprocessing as Observable

input_file = sys.argv[1]

params = Params()
params.setup(input_file) 

E = Observable.EnergyTemperature(params)
E.plot('energy',True)
# E.plot('temperature',False)

SSF = Observable.StaticStructureFactor(params)
SSF.plot()

rdf = Observable.RadialDistributionFunction(params)
rdf.plot()
