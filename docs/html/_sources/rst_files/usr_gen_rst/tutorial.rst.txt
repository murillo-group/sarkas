
Tutorial
========
Here we provide a step-by-step guide to illustrate how to run Sarkas and how to obtain useful quantities from your simulation data once you run is complete. The topics that are listed in this tutorial are as follows

- Saving output data from Sarkas
- Computing the radial distribution function (RDF)
- Computing the velocity autocorrelation function (VACF)
- Computing the self-diffusion coefficient

Problem Set-up
--------------
Let's consider a strongly coupled plasma comprised of 500 copper ions with a number density of :math:`n_i = 6e+22 \frac{N}{cm^3}` at a temperature :math:`1 eV`, the non-dimensional parameters, assuming a Yukawa potential is used for the interactions, the "Potential" section of the yaml.inp file would be set-up as follows

.. csv-table:: Example values for "Potential" section in yaml.inp file
   :header: "Key", "Value"
   :widths: auto

   "type", "Yukwa"
   "algorithm", "PP"
   "kappa", "3.04"
   "Gamma", "129.12"
   "rc", "3.78"

After adding these parameters to the yaml.inp file, we move to the "Control" section of the yaml.inp file to specify the simulation parameters. Supposing that we want a fairly converged RDF and VACF, we could specify the following values

.. csv-table:: Example values for "Control" section in yaml.inp file
   :header: "Key", "Value"
   :widths: auto

   "dt", "0.1"
   "Neq", "1000"
   "Nstep", "30000"
   "BC", "periodic"
   "writeout", "yes"
   "writexyz", "no"
   "dump_step", "10"
   "random_seed", "1 (deprecated)"
   "restart", "no"
   "verbose", "yes"

.. csv-table:: Example values for "Control" section in yaml.inp file
   :header: "Key", "Value"
   :widths: auto

   "dt", "0.1"
   "Neq", "1000"
   "Nstep", "30000"
   "BC", "periodic"
   "writeout", "yes"
   "writexyz", "no"
   "dump_step", "10"
   "random_seed", "1 (deprecated)"
   "restart", "no"
   "verbose", "yes"


Saving Output from Sarkas
-------------------------

