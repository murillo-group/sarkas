"""
Module for plotting observables.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from S_params import Params
from matplotlib.gridspec import GridSpec
import S_postprocessing as Observable

plt.style.use(
    os.path.join(os.path.join(os.getcwd(), 'src'), 'MSUstyle'))
lw, LW = 2, 2
FSZ, fsz = 16, 16

UNITS = [
    {"Energy": 'J',
     "Time": 's',
     "Length": 'm',
     "Charge": 'C',
     "Temperature": 'K',
     "ElectronVolt": 'eV',
     "Mass": 'kg',
     "Magnetic Field": 'T',
     "Current": "A",
     "Power": "erg/s"},
    {"Energy": 'erg',
     "Time": 's',
     "Length": 'cm',
     "Charge": 'esu',
     "Temperature": 'K',
     "ElectronVolt": 'eV',
     "Mass": 'g',
     "Magnetic Field": 'G',
     "Current": "esu/s",
     "Power": "erg/s"}
]

PREFIXES = {
    "Y": 1e24,
    "Z": 1e21,
    "E": 1e18,
    "P": 1e15,
    "T": 1e12,
    "G": 1e9,
    "M": 1e6,
    "k": 1e3,
    "h": 1.0e2,
    "da": 1.0e1,
    "d": 1.0e-1,
    "c": 1.0e-2,
    "m": 1.0e-3,
    r"$\mu$": 1.0e-6,
    "n": 1.0e-9,
    r"$\AA$": 1.0e-10,
    "p": 1.0e-12,
    "f": 1.0e-15,
    "a": 1.0e-18,
    "z": 1.0e-21,
    "y": 1.0e-24
}


def update_label(old_label, exponent_text_old):
    if exponent_text_old == "":
        return old_label

    try:
        units = old_label[old_label.index("[") + 1:old_label.rindex("]")]
    except ValueError:
        units = ""

    label = old_label.replace("[{}]".format(units), "")

    evalue = float(exponent_text_old)
    # find the prefix
    if PREFIXES["y"] < evalue < PREFIXES["Y"]:
        try:
            prefix = list(PREFIXES.keys())[list(PREFIXES.values()).index(evalue)]
            multiplier = 1.0
        except ValueError:
            try:
                prefix = list(PREFIXES.keys())[list(PREFIXES.values()).index(10 * evalue)]
                multiplier = 10.
            except ValueError:
                prefix = list(PREFIXES.keys())[list(PREFIXES.values()).index(100 * evalue)]
                multiplier = 100.

    exponent_text = evalue

    return "{} [{} {}]".format(label, exponent_text, units)


def format_label_string_with_exponent(ax, axis='both'):
    """ Format the label string with the exponent from the ScalarFormatter """
    ax.ticklabel_format(axis=axis, style='sci')

    axes_instances = []
    if axis in ['x', 'both']:
        axes_instances.append(ax.xaxis)
    if axis in ['y', 'both']:
        axes_instances.append(ax.yaxis)

    for ax in axes_instances:
        # ax.major.formatter._useMathText = True
        plt.draw()  # Update the text
        exponent_text = ax.get_offset_text().get_text()
        print(exponent_text)
        label = ax.get_label().get_text()
        ax.offsetText.set_visible(False)
        ax.set_label_text(update_label(label, exponent_text))


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
# E.statistics("Total Energy", max_no_divisions=1000, show=True)
E.boxplot("Total Energy", True)

# clr = [24/255, 69/255, 49/255]
# clr2 = [240/255, 133/255, 33/255]


# E.statistics("Temperature", max_no_divisions=1000, show=True)
# params.Control.Nsteps = 10000
# Z = Observable.VelocityAutocorrelationFunctions(params)
# Z.compute()
# Z.plot()

# params.PostProcessing.dsf_no_ka_values = [5, 5, 5]
# DSF = Observable.DynamicStructureFactor(params)
# DSF.compute()
# DSF.plot(show=False)
# #
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

# TC = Observable.TransportCoefficients(params)
# TC.compute("Electrical Conductivity",show=True)
# TC.compute("Diffusion",show=True)
#
# XYZ = Observable.XYZFile(params)
# XYZ.save()
