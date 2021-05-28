
# Import the usual libraries                                                                                                     
import numpy as np
import os                                                                                                                        
import pandas as pd                                                                                                              
                                                                                                                                 
# plt.style.use('MSUstyle')                                                                                                     
# Import sarkas                                                                                                                 
from sarkas.processes import PreProcess
from sarkas.utilities.timing import SarkasTimer as stimer
meshes = np.array([8, 16, 24, 32, 64, 128], dtype = np.int)                                            
kappas = np.linspace(0.1, 1.0, 19)                                                                                               

data = pd.DataFrame()
# COULOMB
nps = 1000
t0 = stimer.current()
flename = os.path.join('input', "ocp_pppm_N{}k.yaml".format(int(nps*1e-3)) )
preproc = PreProcess(flename)
preproc.setup(read_yaml=True)
max_cells = int(0.5 * preproc.parameters.box_lengths.min() / preproc.parameters.a_ws)
pp_cells = np.arange(3, max_cells, dtype= np.int)
rcs = preproc.parameters.box_lengths[0]/pp_cells
for im, mesh in enumerate(meshes):
    alphas = np.linspace(0.1, 2.0, 29) /preproc.parameters.a_ws
    print('Mesh {}'.format(mesh))
    for ia, alpha in enumerate(alphas):
        for ir, rc in enumerate(rcs):
            for cao in range(7):
                new_args = {"Potential":
                            {"rc": rc,
                             "pppm_mesh": np.ones(3, dtype = np.int)*mesh,
                             "pppm_aliases": [3,3,3],
                             "pppm_cao": cao + 1,
                             "pppm_alpha_ewald": alpha # 1/[m]
                            }
                           }
                preproc = PreProcess(flename)
                preproc.setup(read_yaml=True, other_inputs=new_args)
                preproc.run(timing=True, remove=True)
                data = data.append(
                    {
                        'potential': preproc.potential.type,
                        'num_species': preproc.parameters.num_species,
                        'species_num': preproc.parameters.species_num,
                        'species_charges': preproc.parameters.species_charges,
                        'species_num_dens': preproc.parameters.species_num_dens,
                        'species_masses': preproc.parameters.species_masses,
                        'species_temperatures': preproc.parameters.species_temperatures,
                        'total_num_ptcls': preproc.parameters.total_num_ptcls,
                        'coupling_constant': preproc.parameters.coupling_constant,
                        'a_ws': preproc.parameters.a_ws,
                        'kappa': 0.0,
                        'rc': preproc.potential.rc,
                        'pp_cells': int(preproc.parameters.box_lengths.min()/preproc.potential.rc),
                        'Mx': preproc.potential.pppm_mesh[0],
                        'My': preproc.potential.pppm_mesh[1],
                        'Mz': preproc.potential.pppm_mesh[2],
                        'hx': preproc.potential.pppm_h_array[0],
                        'hy': preproc.potential.pppm_h_array[1],
                        'hz': preproc.potential.pppm_h_array[2],
                        'pppm_alpha_ewald': preproc.potential.pppm_alpha_ewald,
                        'pppm_aliases': preproc.potential.pppm_aliases,
                        'pppm_cao': preproc.potential.pppm_cao,
                        'hx alpha': preproc.potential.pppm_h_array[0] * preproc.potential.pppm_alpha_ewald,
                        'hy alpha': preproc.potential.pppm_h_array[1] * preproc.potential.pppm_alpha_ewald,
                        'hz alpha': preproc.potential.pppm_h_array[2] * preproc.potential.pppm_alpha_ewald,
                        'pppm_pp_error': preproc.parameters.pppm_pp_err,
                        'pppm_pm_error': preproc.parameters.pppm_pm_err,
                        'force error': preproc.parameters.force_error,
                        'pm_acc_time [ns]': np.mean(preproc.pm_acc_time[1:]),
                        'pp_acc_time [ns]': np.mean(preproc.pp_acc_time[1:])

                    },
                    ignore_index=True
                )

data.to_csv('coulomb_dataset_N{}k.csv'.format(int(nps*1e3)), index=False, )
tend = stimer.current()
print(tend-t0)
print('Total time {} hrs {} min {} sec {} msec {} usec {} nsec'.format(*stimer.time_division(tend-t0) ))