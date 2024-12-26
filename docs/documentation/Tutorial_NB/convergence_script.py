# Import the usual libraries
import numpy as np
import matplotlib.pyplot as plt

import os
plt.style.use('MSUstyle')

from multiprocessing import Process

# Import sarkas
from sarkas.processes import PreProcess, Simulation


def run_sarkas_simulation(input_file_name, N):
    args = {'Particles': [
        {"Species" :{
            'name' :'Al',
            'number_density': 6.03e+22,
            'mass': 4.513068464544e-23,
            'Z': 3.0,
            'temperature_eV': 1.0,
            'num': N,
            'replace': True
            }
        }
    ],
    "IO" : {
             'verbose': False,
            'job_dir': f'N{N}'
                }
    }

    # Initialize the Simulation class
    sim = Simulation(input_file_name)
    
    # Setup the simulation's parameters
    sim.setup(read_yaml=True, other_inputs=args)
    # Run the simulation
    sim.run()

    print(f"N = {N} done")

def run_dt_simulation(input_file_name, rc, dt, tau_wp):

    # Need to change the thermostat.
    # tau necessary to decay to 0.01 within 5 plasma periods
    tau_B = - 5.0 * tau_wp/np.log(0.01)/dt

    N_steps = int(20*tau_wp/dt)
    dump_step = int(0.2 * tau_wp/dt)
    N_eq = int(2.0 * tau_wp/dt)
    
    args = {'Particles': [ 
        {"Species" :{
            'name' :'Al',
            'number_density': 6.03e+22,
            'mass': 4.513068464544e-23,
            'Z': 3.0,
            'temperature_eV': 1.0,
            'num': 2000,
            'replace': True
        }
        }
    ],
    "Potential": {"rc" : rc},
    "Integrator" : {
        "dt" : dt,
        "thermalization_timestep": N_eq,
        "berendsen_tau" : tau_B},
    "Parameters" : {
        "equilibration_steps": N_steps,
        "production_steps": N_steps,
        "eq_dump_step": dump_step,
        "prod_dump_step": dump_step,
    },
    
    "IO" : {
             'verbose': False,
            'job_dir': f'dt_{dt/tau_wp:.3f}'
                }
    }
    # Initialize the Simulation class
    sim = Simulation(input_file_name)
    
    # Setup the simulation's parameters
    sim.setup(read_yaml=True, other_inputs=args)
    # Run the simulation
    sim.run()
    print(f"dt = {dt/tau_wp:.3f} done")

if __name__ == '__main__':
    # Create the file path to the YAML input file
    input_file_name = os.path.join('input_files', 'yocp_convergence.yaml' )
    num_particles = np.array([250, 500, 1000, 2000, 5000])

    pre = PreProcess(input_file_name)
    pre.setup(read_yaml=True)
    pre.run()
    
    # processes = []
    # for i, N in enumerate(num_particles):
    #     p0 = Process(target = run_sarkas_simulation, args = (input_file_name, N,))
    #     processes.append(p0)
    #     p0.start()
        
    # for p in processes:
    #     p.join()

    tau_wp = 2.0 * np.pi / pre.species[0].plasma_frequency
    dt_array = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2]) * tau_wp
    rc = 6.0 * pre.parameters.a_ws 
    processes = []
    
    for i, dt in enumerate(dt_array):
        p0 = Process(target = run_dt_simulation, args = (input_file_name, rc, dt, tau_wp,))
        processes.append(p0)
        p0.start()
        
    for p in processes:
        p.join()
