import numpy as np
import numba as nb
import math as mt
import sys
import time

import S_global_names as glb
#import S_update as update

@nb.autojit
#def particle_particle(kappa,G,rc,Lv,pos,acc_s_r):
def particle_particle(pos,acc_s_r):
    rc = glb.rc
    kappa = glb.kappa
    G = glb.G
    ai = glb.ai
   
    N = len(pos[:,0])
    d = len(pos[0,:])

    Lx = glb.Lv[0]
    Ly = glb.Lv[1]
    Lz = glb.Lv[2]

    empty = -50

    Lxd = int(np.floor(Lx/rc))
    Lyd = int(np.floor(Ly/rc))
    Lzd = int(np.floor(Lz/rc))

    rc_x = Lx/Lxd
    rc_y = Ly/Lyd
    rc_z = Lz/Lzd

    Ncell = Lxd*Lyd*Lzd
    
    #print(Ncell)

    head = np.arange(Ncell)
    head.fill(empty)
    ls = np.arange(N)

    rshift = np.zeros(d)

    U_s_r = 0.0
    #count = 0
    #count_rc = 0
    
    acc_s_r.fill(0.0)
    for i in range(N):
    
        cx = int(np.floor(pos[i,0]/rc_x))
        cy = int(np.floor(pos[i,1]/rc_y))
        cz = int(np.floor(pos[i,2]/rc_z))
        c = cx + cy*Lxd + cz*Lxd*Lyd
    
        ls[i] = head[c]
        head[c] = i
    
    for cx in range(Lxd):
        for cy in range(Lyd):
            for cz in range(Lzd):

                c = cx + cy*Lxd + cz*Lxd*Lyd

                for cz_N in range(cz-1,cz+2):
                    for cy_N in range(cy-1,cy+2):
                        for cx_N in range(cx-1,cx+2):
            
                            if (cx_N < 0): 
                                cx_shift = Lxd
                                rshift[0] = -Lx
                            elif (cx_N >= Lxd): 
                                cx_shift = -Lxd
                                rshift[0] = Lx
                            else:
                                cx_shift = 0
                                rshift[0] = 0.0
                
                            if (cy_N < 0): 
                                cy_shift = Lyd
                                rshift[1] = -Ly
                            elif (cy_N >= Lyd): 
                                cy_shift = -Lyd
                                rshift[1] = Ly
                            else:
                                cy_shift = 0
                                rshift[1] = 0.0
                
                            if (cz_N < 0): 
                                cz_shift = Lzd
                                rshift[2] = -Lz
                            elif (cz_N >= Lzd): 
                                cz_shift = -Lzd
                                rshift[2] = Lz
                            else:
                                cz_shift = 0
                                rshift[2] = 0.0
                
                            c_N = (cx_N+cx_shift) + (cy_N+cy_shift)*Lxd + (cz_N+cz_shift)*Lxd*Lyd
            
                            i = head[c]
            
                            while(i != empty):
                
                                j = head[c_N]
                
                                while(j != empty):
                    
                                    if i < j:
                                        dx = pos[i,0] - (pos[j,0] + rshift[0])
                                        dy = pos[i,1] - (pos[j,1] + rshift[1])
                                        dz = pos[i,2] - (pos[j,2] + rshift[2])
                                        #r = np.linalg.norm(rv)  
                                        r = np.sqrt(dx**2 + dy**2 + dz**2)
                                        #count = count + 1
                                        if r < rc:
#                                            U_s_r = 0.
                                            f1 = 0.
                                            f2 = 0.
                                            f3 = 0.
                                            fr = 0.
                                            #########################
                                            if(glb.potential_type == glb.Yukawa_P3M): # P3M. Do not compare strings. It is very expensive!
                                            #Gautham's thesis Eq. 3.22
                                            # Short range  potential
                                                U_s_r = U_s_r + (0.5/r)*(np.exp(kappa*r)*mt.erfc(G*r + 0.5*kappa/G) + np.exp(-kappa*r)*mt.erfc(G*r - 0.5*kappa/G))
                                                f1 = (0.5/r**2)*np.exp(kappa*r)*mt.erfc(G*r + 0.5*kappa/G)*(1-kappa*r)
                                                f2 = (0.5/r**2)*np.exp(-kappa*r)*mt.erfc(G*r - 0.5*kappa/G)*(1+kappa*r)
                                                f3 = (G/np.sqrt(np.pi)/r)*(np.exp(-(G*r + 0.5*kappa/G)**2)*np.exp(kappa*r) + np.exp(-(G*r - 0.5*kappa/G)**2)*np.exp(-kappa*r))
                                                fr = f1+f2+f3

                                            if(glb.potential_type == glb.Yukawa_PP): # PP
                                                U_s_r = U_s_r + np.exp(-kappa*r)/r 
#                                                print("r, U_s_r, kappa = ", r, U_s_r, kappa)
                                                f1 = 1./r**2*np.exp(-kappa*r)
                                                f2 = kappa/r*np.exp(-kappa*r)
                                                fr = f1+f2
                                            if(glb.potential_type == glb.EGS): # PP
                                                phi = 0
                                                if(glb.nu <= 1):
                                                    temp1 = (1+glb.alpha)*np.exp(-r/glb.lambda_m)
                                                    temp2 = (1-glb.alpha)*np.exp(-r/glb.lambda_p)
                                                    phi = 1./(2*r)*(temp1 + temp2)
                                                    fr = phi/r + 1/(2*r)*(temp1/glb.lambda_m + temp2/glb.lambda_p) 
                                                elif(glb.nu > 1):
#                                                    phi =0
                                                    temp1 = np.cos(r/glb.gamma_m)
                                                    temp2 = np.sin(r/glb.gamma_m)
                                                    temp3 = np.exp(-r/glb.gamma_p)
                                                    phi = 1./(r)*(temp1 + glb.alphap*temp2)*temp3
                                                    fr1 = phi/r
                                                    fr2 = - 1/r*(temp2/glb.gamma_m + glb.alphap*temp1/glb.gamma_m)*temp3
                                                    fr3 = phi/glb.gamma_p
                                                    fr = fr1 + fr2 + fr3
                                                U_s_r = U_s_r + phi
                                           ########################## 
                                            acc_s_r[i,0] = acc_s_r[i,0] + fr*dx/r
                                            acc_s_r[i,1] = acc_s_r[i,1] + fr*dy/r
                                            acc_s_r[i,2] = acc_s_r[i,2] + fr*dz/r
                                            
                                            acc_s_r[j,0] = acc_s_r[j,0] - fr*dx/r
                                            acc_s_r[j,1] = acc_s_r[j,1] - fr*dy/r
                                            acc_s_r[j,2] = acc_s_r[j,2] - fr*dz/r
            
                                    j = ls[j]
                    
                                i = ls[i]
    return U_s_r, acc_s_r
