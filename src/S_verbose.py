import numpy as np
from inspect import currentframe, getframeinfo
import time

class Verbose:
    """ 
    Class to handle verbose output to screen.

    Parameters
    ----------
    params : class
        Simulation's parameters

    """
    def __init__(self, params):
        print("Sarkas Ver. 1.0")
        self.params = params

    def sim_setting_summary(self):
        """
        Print out to screen a summary of simulation's parameters.
        """
        params = self.params
        print('\n\n----------- Molecular Dynamics Simulation ----------------------')
        print('No. of particles = ', params.total_num_ptcls)
        print('No. of species = ', len(params.species) )
        print('units: ', params.units)
        print('Temperature = {:2.6e} [K]'.format(params.T_desired) )
        
        print('\nNo. of non-zero box dimensions = ', int(params.d) )
        print('Wigner-Seitz radius = {:2.6e}'.format(params.aws) )
        print('Box length along x axis = {:2.6e} = {:2.6e} a_ws'.format(params.Lv[0],params.Lv[0]/params.aws) )
        print('Box length along y axis = {:2.6e} = {:2.6e} a_ws'.format(params.Lv[1],params.Lv[1]/params.aws) )
        print('Box length along z axis = {:2.6e} = {:2.6e} a_ws'.format(params.Lv[2],params.Lv[2]/params.aws) )
        
        print('\nPotential: ', params.Potential.type)
        
        if (params.Potential.type == 'Yukawa'):
            print('kappa = {:2.2f}'.format( params.Potential.matrix[0, 0, 0]*params.aws) )
            if (len(params.species) > 1):
                print('Gamma_eff = {:4.2f}'.format( params.Potential.Gamma_eff) )
            else:
                print('Gamma = {:4.2f}'.format( params.Potential.matrix[1, 0, 0]) )

        if (params.Potential.type == 'LJ'):
            print('epsilon = {:2.6e}'.format( params.Potential.matrix[0, 0, 0]) )
            print('sigma = {:2.6e}'.format( params.Potential.matrix[1, 0, 0]) )
        
        if (params.Potential.type == "QSP"):
            print("e deBroglie wavelength = {:2.6e} ".format(params.Potential.matrix[0,0,0]/np.sqrt(2.0) ) )
            print("e deBroglie wavelength / a_ws = {:2.6e} ".format(params.Potential.matrix[0,0,0]/(np.sqrt(2.0)*params.ai) ) )
            print("ion deBroglie wavelength = {:2.6e} ".format(params.Potential.matrix[0,1,1]/np.sqrt(2.0) ) )
            print("ion deBroglie wavelength / a_ws = {:2.6e} ".format(params.Potential.matrix[0,1,1]/np.sqrt(2.0)/params.ai ) )
            print("e-i Coupling Parameter = {:3.3f} ".format(abs(params.Potential.matrix[1,0,1])/(params.ai*params.kB*params.Ti) ) )
            print("rs Coupling Parameter = {:3.3f} ".format(params.rs) )

        if (params.Magnetic.on):
            print('\nMagnetized Plasma')
            for ic in range(params.num_species):
                print('Cyclotron frequency of Species {:2} = {:2.6e}'.format( ic + 1, params.species[ic].omega_c) )
                print('beta_c of Species {:2} = {:2.6e}'.format( ic + 1, params.species[ic].omega_c/params.species[ic].wp) )

        print("\nAlgorithm = ", params.Potential.method )
        if (params.Potential.method == 'P3M'):
            print('Ewald_parameter * a_ws = {:2.6e}'.format(params.Potential.matrix[-1,0,0]*params.aws) )
            print('Grid_size * Ewald_parameter (h * alpha) = {:2.6e}'.format(params.P3M.hx*params.P3M.G_ew) )
            print('rcut/a_ws = {:2.6e}'.format(params.Potential.rc/params.aws) )
            print('Mesh = ', params.P3M.MGrid)
            print('PM Force Error = {:2.6e}'.format(params.P3M.PM_err) )
            print('No. of cells per dimension = {:2}'.format( int(params.L/params.Potential.rc) ) ) 
            print('No. of neighbors per particle = {:6}'.format( int(params.total_num_ptcls*(params.Potential.rc/params.L)**(3.0)  ) ) )
            print('PP Force Error = {:2.6e}'.format(params.P3M.PP_err) )
            print('Tot Force Error = {:2.6e}'.format(params.P3M.F_err) )
        else:
            print('rcut/a_ws = {:2.6e}'.format(params.Potential.rc/params.aws) )
            print('No. of cells per dimension = {:2}'.format( int(params.L/params.Potential.rc) ) ) 
            print('No. of neighbors per particle = {:6}'.format( int(params.total_num_ptcls*(params.Potential.rc/params.L)**(3.0)  ) ) )
            print('PP Force Error = {:2.6e}'.format(params.PP_err) )

        print('\ntime step = {:2.6e} [s]'.format(params.Control.dt ) )
        if (params.Potential.type == 'Yukawa'):
            print('ion plasma frequency = {:2.6e} [Hz]'.format(params.wp) )
            print('wp dt = {:2.4f}'.format( params.Control.dt*params.species[0].wp ) )  

        if (params.Potential.type == 'QSP'):
            print('e plasma frequency = {:2.6e} [Hz]'.format(params.species[0].wp) )
            print('ion plasma frequency = {:2.6e} [Hz]'.format(params.species[1].wp) )
            print('wp dt = {:2.4f}'.format( params.Control.dt*params.species[0].wp ) )  

        print('\nNo. of equilibration steps = ', params.Control.Neq)
        print('No. of post-equilibration steps = ', params.Control.Nt)
        print('snapshot interval = ', params.Control.dump_step)
        print('Periodic boundary condition {1=yes, 0=no} =', params.Control.PBC)

        if (params.Langevin.on):
            print('Langevin model = ', params.Langevin.type)

        print('\nSmallest interval in Fourier space for S(q,w): qa_min = {:2.6e}'.format( 2.0*np.pi*params.aws/params.Lx) )

    def time_stamp(self, time_stamp):
        """
        Print out to screen elapsed times.

        Parameters
        ----------
        time_stamp : array
            Array of time stamps.

        """
        t = time_stamp
        #t[1] - t[0] is the time to read params
        init_hrs = int( (t[2] - t[0])/3600 )
        init_min = int( (t[2] - t[0] - init_hrs*3600)/60 )
        init_sec = int( (t[2] - t[0] - init_hrs*3600 - init_min*60) )
        print('Time for initialization = {} hrs {} mins {} secs'.format( init_hrs, init_min, init_sec ) )

        eq_hrs = int( (t[3] - t[2])/3600 )
        eq_min = int( (t[3] - t[2] - eq_hrs*3600)/60 )
        eq_sec = int( (t[3] - t[2] - eq_hrs*3600 - eq_min*60) )
        print('Time for equilibration = {} hrs {} mins {} secs'.format( eq_hrs, eq_min, eq_sec ) )
        if (self.params.Magnetic.on and self.params.Magnetic.elec_therm):
            mag_eq_hrs = int( (t[4] - t[3])/3600 )
            mag_eq_min = int( (t[4] - t[3] - mag_eq_hrs*3600)/60 )
            mag_eq_sec = int( (t[4] - t[3] - mag_eq_hrs*3600 - mag_eq_min*60) )
            print('Time for magnetic equilibration = {} hrs {} mins {} secs'.format( mag_eq_hrs, mag_eq_min, mag_eq_sec ) )

            prod_hrs = int( (t[5] - t[4])/3600 )
            prod_min = int( (t[5] - t[4] - prod_hrs*3600)/60 )
            prod_sec = int( (t[5] - t[4] - prod_hrs*3600 - prod_min*60) )
            print('Time for production = {} hrs {} mins {} secs'.format( prod_hrs, prod_min, prod_sec ) )
            
            tot_hrs = int( (t[6] - t[0])/3600 )
            tot_min = int( (t[6] - t[0] - tot_hrs*3600)/60 )
            tot_sec = int( (t[6] - t[0] - tot_hrs*3600 - tot_min*60) )
            
            print('Total elapsed time = {} hrs {} mins {} secs'.format( tot_hrs, tot_min, tot_sec ) )
        
        else:

            prod_hrs = int( (t[4] - t[3])/3600 )
            prod_min = int( (t[4] - t[3] - prod_hrs*3600)/60 )
            prod_sec = int( (t[4] - t[3] - prod_hrs*3600 - prod_min*60) )
            print('Time for production = {} hrs {} mins {} secs'.format( prod_hrs, prod_min, prod_sec ) )
            
            tot_hrs = int( (t[4] - t[0])/3600 )
            tot_min = int( (t[4] - t[0] - tot_hrs*3600)/60 )
            tot_sec = int( (t[4] - t[0] - tot_hrs*3600 - tot_min*60) )
            
            print('Total elapsed time = {} hrs {} mins {} secs'.format( tot_hrs, tot_min, tot_sec ) )
