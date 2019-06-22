from mpi4py import MPI
import numpy as np
import numba as nb
import math as mt
import sys
import time

import S_global_names as glb
#import S_update as update

global DEBUG
DEBUG = False

#@nb.autojit
#def particle_particle(kappa,G,rc,Lv,pos,acc_s_r):
def particle_particle(pos, vel, acc_s_r, mpiComm):
    rc = glb.rc
    kappa = glb.kappa
    G = glb.G
    ai = glb.ai

    #N = len(pos[:,0])
    d = len(pos[0,:])

    #Lx = glb.Lv[0]
    #Ly = glb.Lv[1]
    #Lz = glb.Lv[2]
    Lx = mpiComm.Ll[0]
    Ly = mpiComm.Ll[1]
    Lz = mpiComm.Ll[2]

    empty = -50
    emptyArray = np.array([-50.,-50.,-50.,-50.,-50.,-50.])

    Lxd = int(np.floor(Lx/rc))
    Lyd = int(np.floor(Ly/rc))
    Lzd = int(np.floor(Lz/rc))

    rc_x = Lx/Lxd
    rc_y = Ly/Lyd
    rc_z = Lz/Lzd

    Ncell = Lxd*Lyd*Lzd

    head = np.arange(Ncell)
    head.fill(empty)
    #ls = np.arange(N)

    rshift = np.zeros(d)

    #=== Particle Migration-SEND =============

    s = MPI.Status()
    Lmax = mpiComm.Lmax
    Lmin = mpiComm.Lmin
    LocalL = mpiComm.Ll
    size = mpiComm.size

    posX = pos[:,0]
    posY = pos[:,1]
    posZ = pos[:,2]
    migFilter_X = np.logical_or(posX > Lmax[0] , posX < Lmin[0] )
    migFilter_Y = np.logical_or(posY > Lmax[1] , posY < Lmin[1] )
    migFilter_Z = np.logical_or(posZ > Lmax[2] , posZ < Lmin[2] )
    migFilter = np.logical_or.reduce( (migFilter_X,migFilter_Y,migFilter_Z) )


    migIndex = np.where(migFilter)
    migBuff_pos = pos[migIndex[0],:]
    migBuff_vel = vel[migIndex[0],:]
    migBuff = np.stack((migBuff_pos,migBuff_vel),axis=2)
    pos = np.delete(pos,migIndex[0],0)
    vel = np.delete(vel,migIndex[0],0)
    N = len(pos[:,0])
    NsP = len(migBuff_pos)
    NsV = len(migBuff_vel)

    buffLen = 600
    sendDict = {}
    for i in range(len(migBuff[:,0])):
        rank = mpiComm.posToRank(migBuff[i,:,0])
        if rank in sendDict:
            index = int(sendDict[rank][0])
            if index >= len(sendDict[rank]):
                Ln = len(sendDict[rank])
                Ln *= 6
                temp = np.ones( Ln )
                temp[:,0:index] = sendDict[rank]
                sendDict[rank] = temp
            sendDict[rank][index:index+6] = migBuff[i,:,:].flatten()
            sendDict[rank][0] = index+6
        else:
            sendDict[rank] = np.ones( buffLen )
            sendDict[rank][1:7] = migBuff[i,:,:].flatten()
            sendDict[rank][0]=7

    send_requests=[]
    for i in mpiComm.neigs.ranks:
        if i in sendDict:
            if mpiComm.rank==0:
                print("exchange!!")
            Sreq = mpiComm.comm.Isend([sendDict[i][1:int(sendDict[i][0])], MPI.DOUBLE], dest = i, tag = mpiComm.rank)
            send_requests.append(Sreq)
        else:
            Sreq = mpiComm.comm.Isend([emptyArray, MPI.DOUBLE], dest = i,  tag = mpiComm.rank)
            send_requests.append(Sreq)


    #====================================

    #LCL for particles that DID NOT migrate
    ls = np.arange(N)
    U_s_r = 0.0
    for i in range(N):

        cx = int(np.floor((pos[i,0] - mpiComm.Lmin[0])/rc_x))
        cy = int(np.floor((pos[i,1] - mpiComm.Lmin[1])/rc_y))
        cz = int(np.floor((pos[i,2] - mpiComm.Lmin[2])/rc_z))
        c = cx + cy*Lxd + cz*Lxd*Lyd

        ls[i] = head[c]
        head[c] = i


    #=== Particle Migration-Recv =============

    recvB = []
    for i in mpiComm.neigs.ranks:
        probS = mpiComm.comm.Probe(status=s,source=i,tag=i)
        count = s.Get_count(datatype=MPI.DOUBLE)
        recvB.append(np.empty(count,dtype=np.float64))

    recv_requests=[]
    p = 0
    for i in mpiComm.neigs.ranks:
        Rreq = mpiComm.comm.Irecv([recvB[p], MPI.DOUBLE], source = i, tag = i)
        recv_requests.append(Rreq)
        p+=1
    recvPos = np.concatenate(recvB,axis=0)

    MPI.Request.waitall(send_requests)
    MPI.Request.waitall(recv_requests)

    recvBuff = recvPos.reshape(int(len(recvPos)/6),3,2)
    recvBuff = recvBuff[np.where(recvBuff[:,0,0]!=-50.)]
    Nr = recvBuff.shape[0]
    if Nr != 0:
        recvPos = recvBuff[:,:,0]
        recvVel = recvBuff[:,:,1]
        vel = np.append(vel,recvVel,0)
        pos = np.append(pos,recvPos,0)
    acc_s_r = np.zeros_like(pos)


    #=========================================

    #LCL for particles that DID migrate
    ls = np.append(ls,np.zeros(Nr,dtype=np.int))
    for i in range(Nr):

        cx = int(np.floor((pos[N+i,0] - mpiComm.Lmin[0])/rc_x))
        cy = int(np.floor((pos[N+i,1] - mpiComm.Lmin[1])/rc_y))
        cz = int(np.floor((pos[N+i,2] - mpiComm.Lmin[2])/rc_z))
        c = cx + cy*Lxd + cz*Lxd*Lyd

        ls[N+i] = head[c]
        head[c] = N+i

    N = len(pos[:,0])

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
                                        r = np.sqrt(dx**2 + dy**2 + dz**2)

                                        if r < rc:
                                            #if mpiComm.rank==0:
                                            #    print("r = ", r)
                                            #    print("pos_i = ",pos[i,:])
                                            #    print("pos_j = ",pos[j,:])
                                            #U_s_r = 0.
                                            #f1 = 0.
                                            #f2 = 0.
                                            #f3 = 0.
                                            #fr = 0.
                                            #########################
                                            #if(glb.potential_type == glb.Yukawa_P3M): # P3M. Do not compare strings. It is very expensive!
                                            #Gautham's thesis Eq. 3.22
                                            # Short range  potential
                                            #    U_s_r = U_s_r + (0.5/r)*(np.exp(kappa*r)*mt.erfc(G*r + 0.5*kappa/G) + np.exp(-kappa*r)*mt.erfc(G*r - 0.5*kappa/G))
                                            #    f1 = (0.5/r**2)*np.exp(kappa*r)*mt.erfc(G*r + 0.5*kappa/G)*(1-kappa*r)
                                            #    f2 = (0.5/r**2)*np.exp(-kappa*r)*mt.erfc(G*r - 0.5*kappa/G)*(1+kappa*r)
                                            #    f3 = (G/np.sqrt(np.pi)/r)*(np.exp(-(G*r + 0.5*kappa/G)**2)*np.exp(kappa*r) + np.exp(-(G*r - 0.5*kappa/G)**2)*np.exp(-kappa*r))
                                            #    fr = f1+f2+f3

                                            #if(glb.potential_type == glb.Yukawa_PP): # PP
                                            U_s_r = U_s_r + np.exp(-kappa*r)/r
                                            f1 = 1./r**2*np.exp(-kappa*r)
                                            f2 = kappa/r*np.exp(-kappa*r)
                                            fr = f1+f2
                                           ##########################
                                            acc_s_r[i,0] = acc_s_r[i,0] + fr*dx/r
                                            acc_s_r[i,1] = acc_s_r[i,1] + fr*dy/r
                                            acc_s_r[i,2] = acc_s_r[i,2] + fr*dz/r

                                            acc_s_r[j,0] = acc_s_r[j,0] - fr*dx/r
                                            acc_s_r[j,1] = acc_s_r[j,1] - fr*dy/r
                                            acc_s_r[j,2] = acc_s_r[j,2] - fr*dz/r

                                    j = ls[j]

                                i = ls[i]
    return U_s_r, acc_s_r, vel,pos
