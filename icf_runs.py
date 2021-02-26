import numpy as np
from sarkas.simulation.params import Params
from sarkas.simulation import simulation
#import sarkas.tools.testing as Testing
from sarkas.tools.postprocessing import (Thermodynamics,
                                         VelocityAutocorrelationFunctions,
                                         ElectricCurrent,
                                         StaticStructureFactor,
                                         Transport, read_pickle)

#from sarkas.tools.postprocessing import read_pickle

args = dict()
args["input_file"] =  "sarkas/examples/yukawa_icf_cgs.yaml"

job_id = 0
rnd = 14987516

rg = np.random.Generator(np.random.PCG64(rnd) )

for i in range(job_id, job_id + 1):
    # Save each simulation's data into own its directory
    args["job_dir"] = "icf_sim_" + str(i)
    args["job_id"] = "icf_sim_" + str(i)
    args["seed"] = rg.integers(0, high=123456789)
    args["verbose"] = True

    # sim_dir = os.path.join('Simulations', args["job_dir"])
    
    params = Params()
    params.setup(args)
    params.rand_seed = args["seed"]
    #params = read_pickle('Simulations/icf_sim_0')
    #Testing.main(params)    
    simulation.run(params)
    
    # # Plot energy
    E = Thermodynamics(params)
    E.compute_pressure_quantities()
    E.temp_energy_plot(params, phase = 'equilibration', show =False)
    E.temp_energy_plot(params, phase = 'production', show =False)

    J = ElectricCurrent(params)
    J.plot()

    VACF = VelocityAutocorrelationFunctions(params)  
    VACF.compute()
    VACF.plot(intercurrent=False)
    VACF.plot(intercurrent=True)
    
    SSF = StaticStructureFactor(params)
    SSF.compute()
    SSF.plot(errorbars=False)

    # sigma = Transport.compute(params, "Electrical Conductivity", True)
    # diff = Transport.compute(params, "Diffusion", True)
    # eta = Transport.compute(params, "Viscosity", True)
    # if i == job_id:
    #     Tot_Sk = np.zeros( (SSF.dataframe["ka values"].shape[0], SSF.no_Sk)) 
    #     ka_values = np.array(SSF.dataframe["ka values"])
        
    #     Tot_vacf = np.zeros( (VACF.no_dumps, VACF.no_vacf))
    #     time = np.array(VACF.dataframe["Time"])

    # sp_indx = 0
    # for sp_i in range(SSF.no_species):
    #     for sp_j in range(sp_i, SSF.no_species):
    #         column = "{}-{} SSF".format(SSF.species_names[sp_i], SSF.species_names[sp_j])

    #         Tot_Sk[:, sp_indx] += 0.3333 * np.array(SSF.dataframe[column])
            
    #         sp_indx += 1
    
    # v_ij = 0
    # for sp in range(VACF.no_species):
    #     Tot_vacf[:, v_ij] =  VACF.dataframe["{} Total Velocity ACF".format(VACF.species_names[sp])] 
    #     for sp2 in range(sp + 1, VACF.no_species):
    #         v_ij += 1
    #         Tot_vacf[:, v_ij] = VACF.dataframe["{}-{} Total Current ACF".format(VACF.species_names[sp],
    #                                                         VACF.species_names[sp2])] 

    print("Job: {}, Run {} out of 3 completed".format(job_id, i) )
    del params, E, J, SSF, VACF, #sigma, diff, eta

# fig, ax = plt.subplots(1,1)
# fig2, ax2 = plt.subplots(1,1)

# sp_indx = 0
# for sp_i in range(SSF.no_species):
#     for sp_j in range(sp_i, SSF.no_species):
#         column = "{}-{}".format(SSF.species_names[sp_i], SSF.species_names[sp_j])

#         ax.plot(ka_values, Tot_Sk[:,sp_indx], label = column)
#         ax2.plot(time, Tot_vacf[:,sp_indx]/Tot_vacf[0, sp_indx], label = column)

#         sp_indx += 1

# ax.set_title('Static Structure Factor')
# ax.set_xlabel(r'ka')
# fig.savefig('SSF_' + str(job_id) + '.png') 

# ax2.set_title('VACF')
# ax2.set_xscale('log')
# ax2.set_xlabel(r'Time')
# fig2.savefig('VACF_' + str(job_id) + '.png')
