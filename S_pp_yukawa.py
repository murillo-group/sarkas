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
def particle_particle(pos,acc_s_r, vel, mpiComm):
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

    #if mpiComm.rank==0:
    #    print("rank 0 N=",len(pos[:,0]))
    #    print("Lxd = ",Lxd)
    #    print("Lyd = ",Lyd)
    #    print("Lzd = ",Lzd)

    #if mpiComm.rank==1:
    #    print("rank 1 N=",len(pos[:,0]))
    #    print(np.where(pos[:,0]>9.42698613))
    #    print(np.where(pos[:,1]>9.42698613))
    #    print(np.where(pos[:,2]>9.42698613))

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
    migFilter = np.logical_or(np.logical_or(migFilter_X,migFilter_Y),migFilter_Z)

    migIndex = np.where(migFilter)
    migBuff_pos = pos[migIndex[0],:]
    migBuff_vel = vel[migIndex[0],:]
    migBuff = np.stack((migBuff_pos,migBuff_vel),axis=2)
    pos = np.delete(pos,migIndex[0],0)
    vel = np.delete(vel,migIndex[0],0)

    N = len(pos[:,0])
    #if mpiComm.rank==0:
    #    print(pos)
    #if mpiComm.rank==1:
    #    print(migBuff_pos)

    buffLen = 600
    sendBuff = np.ones( (size,buffLen+1))
    for i in range(len(migBuff[:,0])):
        rank = mpiComm.posToRank(migBuff[i,:,0])
        buff = sendBuff[rank,:]
        index = int(buff[0])
        if index >= sendBuff.shape[1]:
            buffLen *= 6
            temp = np.ones( (mpiComm.size,buffLen+1))
            temp[:,0:index] = sendBuff
            sendBuff = temp
            buff = sendBuff[rank,:]
        buff[index:index+6] = migBuff[i,:].flatten()
        buff[0] = index+6

    #sendDict = {}
    #for i in range(len(migBuff[:,0])):
    #    rank = mpiComm.posToRank(migBuff[i,:])
    #    if rank in sendDict:
    #        sendDict[rank] = np.append(sendDict[rank],migBuff[i,:],axis=0)
    #    else:
    #        sendDict[rank] = migBuff[i,:]
    #
    #print(mpiComm.glbIndexToRank( mpiComm.posToGlb([6.49114691, 7.39617972, 3.29489686])) )
    #print(mpiComm.posToGlb([2.49114691, 7.39617972, 3.29489686]))
    #send_requests=[]
    #for i in range(mpiComm.size):
    #    if i in sendDict:
    #        Sreq = mpiComm.comm.Isend([sendDict[i].flatten(), MPI.DOUBLE], dest = i, tag = mpiComm.rank)
    #        send_requests.append(Sreq)
    #    else:
    #        Sreq = mpiComm.comm.Isend([np.array([-50.0]), MPI.DOUBLE], dest = i,  tag = mpiComm.rank)
    #        send_requests.append(Sreq)

    SIZE = 0
    send_requests=[]
    for i in range(size):
        index = int(sendBuff[i,0])
        if index == 1 and i != mpiComm.rank:
            Sreq = mpiComm.comm.Isend([ np.array([-50.,-50.,-50.]), MPI.DOUBLE], dest = i, tag = mpiComm.rank)
            send_requests.append(Sreq)
        elif i != mpiComm.rank:
            buff = sendBuff[i,1:index]
            SIZE += len(buff)
            Sreq = mpiComm.comm.Isend([buff, MPI.DOUBLE], dest = i, tag = mpiComm.rank)
            send_requests.append(Sreq)
    #if DEBUG:
    #    if(mpiComm.rank==0):
    #        print("send size = ",SIZE)


    #====================================

    #LCL for particles that DID NOT migrate
    ls = np.arange(N)
    U_s_r = 0.0
    #if mpiComm.rank==1:
    #    print(N)
    for i in range(N):

        cx = int(np.floor((pos[i,0] - mpiComm.Lmin[0])/rc_x))
        cy = int(np.floor((pos[i,1] - mpiComm.Lmin[1])/rc_y))
        cz = int(np.floor((pos[i,2] - mpiComm.Lmin[2])/rc_z))
        c = cx + cy*Lxd + cz*Lxd*Lyd
        #if mpiComm.rank==0:
        #    print("cx = ",cx)
        #    print("cy = ",cy)
        #    print("cz = ",cz)
        #    print("c = ",c)
        #    print(pos[i,:])

        ls[i] = head[c]
        head[c] = i

    #=== Particle Migration-Recv =============

    buffSize = 0
    for p in range(size):
        if p != mpiComm.rank:
            probS = mpiComm.comm.Probe(status=s,source=p,tag=p)
            buffSize += s.Get_count(datatype=MPI.DOUBLE)

    #if DEBUG:
    #    if mpiComm.rank==1:
    #        print("recv size = ",buffSize)


    recv_requests=[]
    recvPos = np.empty(buffSize,dtype=np.float64)
    for p in range(size):
        if p != mpiComm.rank:
            Rreq = mpiComm.comm.Irecv([recvPos, MPI.DOUBLE], source = p, tag = p)
            recv_requests.append(Rreq)

    MPI.Request.waitall(send_requests)
    MPI.Request.waitall(recv_requests)


    Nr = int(len(recvPos)/6)
    if Nr != 0:
        recvBuff = recvPos.reshape(Nr,3,2) #breaks if Nr is zero!!
        recvPos = recvBuff[:,:,0]
        recvVel = recvBuff[:,:,1]
        vel = np.append(vel,recvVel,0)
        pos = np.append(pos,recvPos,0)
        acc_s_r = np.zeros_like(pos)


    #=========================================

    #LCL for particles that DID migrate
    # I think this is where the problem is!!
    # if you commment this out then it works
    #if mpiComm.rank==1:
    #    print(Nr)
    ls = np.append(ls,np.zeros(Nr,dtype=np.int))
    for i in range(Nr):

        cx = int(np.floor((pos[N+i,0] - mpiComm.Lmin[0])/rc_x))
        cy = int(np.floor((pos[N+i,1] - mpiComm.Lmin[1])/rc_y))
        cz = int(np.floor((pos[N+i,2] - mpiComm.Lmin[2])/rc_z))
        c = cx + cy*Lxd + cz*Lxd*Lyd

        ls[N+i] = head[c]
        head[c] = N+i

    N = len(pos[:,0])

    #if mpiComm.rank == 0:
    #    print(pos)
    #    print(ls)
    #    print(head)
    #    print(test)
    #if mpiComm.rank==0:
    #    print(pos[19])
    #    print(pos[70])
    #    print(recvPos)
    #    print(pos)

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
                            #if mpiComm.rank == 0:
                            #    print("c_N = ", c_N)
                            #    print("c = ", c)

                            i = head[c]

                            while(i != empty):

                                j = head[c_N]

                                while(j != empty):

                                    if i < j:
                                        dx = pos[i,0] - (pos[j,0] + rshift[0])
                                        dy = pos[i,1] - (pos[j,1] + rshift[1])
                                        dz = pos[i,2] - (pos[j,2] + rshift[2])
                                        r = np.sqrt(dx**2 + dy**2 + dz**2)
                                        #if mpiComm.rank==0:
                                            #print("r_i="+str(i)+"_j="+str(j)+" = ",r)
                                            #print("j = ",j)

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
