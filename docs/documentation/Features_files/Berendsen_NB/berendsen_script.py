# Import the usual libraries
import numpy as np
import matplotlib.pyplot as plt

import os
plt.style.use('MSUstyle')

from multiprocessing import Process

# Import sarkas
from sarkas.processes import PreProcess, Simulation

def run_simulation_berendsen(input_file_name, tau, eq_steps, cycles):

    args = {
        'Integrator': {'thermalization_timestep': 0, # Timesteps before turning on the thermostat
                       'berendsen_tau': tau}, # Change tau for each simulation
        "Parameters":{ "rand_seed": 123456,
                          "equilibration_steps": eq_steps},
         "IO":   # Store all simulations' data in simulations_dir,
                # but save the dumps in different subfolders (job_dir)
            {
                "job_dir": f"tau_cycles{cycles:.2f}",
                "verbose": False # This is so not to print to screen for every run
            },
    }

    sim = Simulation(input_file_name)
    sim.setup(read_yaml=True, other_inputs=args)
    sim.run()
    
    print(f'tau cycle = {cycles:.2f} Done')

def run_simulation_langevin(input_file_name, gamma, eq_steps, cycles):
    
    args = {
        'Integrator': {'langevin_gamma': gamma}, # Change tau for each simulation
        "Parameters":{ "rand_seed": 123456,
                      "equilibration_steps": eq_steps},
        "IO":   # Store all simulations' data in simulations_dir,
                # but save the dumps in different subfolders (job_dir)
            {
                "job_dir": f"gamma_cycles{cycles:.2f}",
                "verbose": False # This is so not to print to screen for every run
            },
    }

    sim = Simulation(input_file_name)
    sim.setup(read_yaml=True, other_inputs=args)
    sim.run()
    
    print(f'gamma cycle = {cycles:.2f} Done')


if __name__ == '__main__':
    # Create the file path to the YAML input file
    input_file_name = os.path.join('input_files', 'yocp_quickstart.yaml' )

    cycles = np.array([0.01, 0.1, 0.2, 1.0, 5.0, 10.0])

    pre = PreProcess(input_file_name)
    pre.setup(read_yaml=True)

    tau_p = 2.0 * np.pi /pre.parameters.total_plasma_frequency

    taus = - tau_p / np.log(0.01) * cycles / pre.parameters.dt
    eq_steps = np.rint(2.0 * cycles * tau_p/ pre.parameters.dt).astype(int)

    processes = []
    # Create the file path to the YAML input file
    input_file_name = os.path.join('input_files', 'yocp_quickstart.yaml' )

    for i, tau in enumerate(taus):

        p0 = Process(target = run_simulation_berendsen, args = (input_file_name, tau, eq_steps[i], cycles[i],) )
        processes.append(p0)
        p0.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    processes = []
    # Create the file path to the YAML input file
    input_file_name = os.path.join('input_files', 'yocp_quickstart_langevin.yaml' )

    pre = PreProcess(input_file_name)
    pre.setup(read_yaml=True)

    tau_p = 2.0 * np.pi /pre.parameters.total_plasma_frequency
    
    gammas = - np.log(0.01)/(2.0 * tau_p * cycles)
    for i, gamma in enumerate(gammas):    
        p0 = Process(target = run_simulation_langevin, args = (input_file_name, gamma, eq_steps[i], cycles[i],) )
        processes.append(p0)
        p0.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()