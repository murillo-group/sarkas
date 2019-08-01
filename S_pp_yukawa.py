from mpi4py import MPI
import numpy as np
import numba as nb
import math as mt
import sys
import time

import S_global_names as glb
#import S_update as update


#def particle_particle(kappa,G,rc,Lv,pos,acc_s_r):
#@nb.jit(nopython=True)
def particle_particle(pos, vel, acc_s_r, mpiComm):
    rc = glb.rc
    kappa = glb.kappa
    G = glb.G
    ai = glb.ai

    d = len(pos[0,:])

    Lx = mpiComm.Ll[0]
    Ly = mpiComm.Ll[1]
    Lz = mpiComm.Ll[2]

    empty = -50
    emptyArray = np.array([-50.,-50.,-50.,-50.,-50.,-50.])

    Lxd = int(np.floor(Lx/rc))
    Lyd = int(np.floor(Ly/rc))
    Lzd = int(np.floor(Lz/rc))

    exterior = mpiComm.exteriorCells

    rc_x = Lx/Lxd
    rc_y = Ly/Lyd
    rc_z = Lz/Lzd

    Ncell = Lxd*Lyd*Lzd

    head = np.arange(Ncell)
    head.fill(empty)

    #====LCL-Copy-Buffers=====================

    Lower = Lzd-1
    Upper = 0
    South = Lyd-1
    North = 0
    East  = Lxd-1
    West  = 0

    U_buff   = np.array([-50.,-50.,-50.],dtype=np.float64)
    UN_buff  = np.array([-50.,-50.,-50.],dtype=np.float64)
    US_buff  = np.array([-50.,-50.,-50.],dtype=np.float64)
    UE_buff  = np.array([-50.,-50.,-50.],dtype=np.float64)
    UW_buff  = np.array([-50.,-50.,-50.],dtype=np.float64)
    UNE_buff = np.array([-50.,-50.,-50.],dtype=np.float64)
    UNW_buff = np.array([-50.,-50.,-50.],dtype=np.float64)
    USE_buff = np.array([-50.,-50.,-50.],dtype=np.float64)
    USW_buff = np.array([-50.,-50.,-50.],dtype=np.float64)

    N_buff  = np.array([-50.,-50.,-50.],dtype=np.float64)
    S_buff  = np.array([-50.,-50.,-50.],dtype=np.float64)
    E_buff  = np.array([-50.,-50.,-50.],dtype=np.float64)
    W_buff  = np.array([-50.,-50.,-50.],dtype=np.float64)
    NE_buff = np.array([-50.,-50.,-50.],dtype=np.float64)
    NW_buff = np.array([-50.,-50.,-50.],dtype=np.float64)
    SE_buff = np.array([-50.,-50.,-50.],dtype=np.float64)
    SW_buff = np.array([-50.,-50.,-50.],dtype=np.float64)

    D_buff   = np.array([-50.,-50.,-50.],dtype=np.float64)
    DN_buff  = np.array([-50.,-50.,-50.],dtype=np.float64)
    DS_buff  = np.array([-50.,-50.,-50.],dtype=np.float64)
    DE_buff  = np.array([-50.,-50.,-50.],dtype=np.float64)
    DW_buff  = np.array([-50.,-50.,-50.],dtype=np.float64)
    DNE_buff = np.array([-50.,-50.,-50.],dtype=np.float64)
    DNW_buff = np.array([-50.,-50.,-50.],dtype=np.float64)
    DSE_buff = np.array([-50.,-50.,-50.],dtype=np.float64)
    DSW_buff = np.array([-50.,-50.,-50.],dtype=np.float64)

    U_shift = mpiComm.neigs.U_shift
    UN_shift = mpiComm.neigs.UN_shift
    US_shift = mpiComm.neigs.US_shift
    UE_shift = mpiComm.neigs.UE_shift
    UW_shift = mpiComm.neigs.UW_shift
    UNE_shift = mpiComm.neigs.UNE_shift
    UNW_shift = mpiComm.neigs.UNW_shift
    USE_shift = mpiComm.neigs.USE_shift
    USW_shift = mpiComm.neigs.USW_shift

    N_shift = mpiComm.neigs.N_shift
    S_shift = mpiComm.neigs.S_shift
    E_shift = mpiComm.neigs.E_shift
    W_shift = mpiComm.neigs.W_shift
    NE_shift = mpiComm.neigs.NE_shift
    NW_shift = mpiComm.neigs.NW_shift
    SE_shift = mpiComm.neigs.SE_shift
    SW_shift = mpiComm.neigs.SW_shift

    D_shift = mpiComm.neigs.D_shift
    DN_shift = mpiComm.neigs.DN_shift
    DS_shift = mpiComm.neigs.DS_shift
    DE_shift = mpiComm.neigs.DE_shift
    DW_shift = mpiComm.neigs.DW_shift
    DNE_shift = mpiComm.neigs.DNE_shift
    DNW_shift = mpiComm.neigs.DNW_shift
    DSE_shift = mpiComm.neigs.DSE_shift
    DSW_shift = mpiComm.neigs.DSW_shift


    #=== Particle Migration-Send =============

    s = MPI.Status()
    Lmax = mpiComm.Lmax
    Lmin = mpiComm.Lmin
    LocalL = mpiComm.Ll
    size = mpiComm.size

    #build Boolean Mask For paticles that
    #will be migrated to neighboring MPI ranks.
    posX = pos[:,0]
    posY = pos[:,1]
    posZ = pos[:,2]
    migFilter_X = np.logical_or(posX > Lmax[0] , posX < Lmin[0] )
    migFilter_Y = np.logical_or(posY > Lmax[1] , posY < Lmin[1] )
    migFilter_Z = np.logical_or(posZ > Lmax[2] , posZ < Lmin[2] )
    migFilter = np.logical_or.reduce( (migFilter_X,migFilter_Y,migFilter_Z) )

    #Create Buffer for particles that will
    #be migrated
    migIndex = np.where(migFilter)
    migBuff_pos = pos[migIndex[0],:]
    migBuff_vel = vel[migIndex[0],:]
    migBuff = np.stack((migBuff_pos,migBuff_vel),axis=2)
    pos = np.delete(pos,migIndex[0],0)
    vel = np.delete(vel,migIndex[0],0)
    N = len(pos[:,0])

    #BuckSort particles to ranks they migrated to
    #buffLen = 600 #needs to be multiple of 6
    #sendDict = {}
    #for i in range(len(migBuff[:,0])):
    #    rank = mpiComm.posToRank(migBuff[i,:,0])
    #    if rank in sendDict:
    #        index = int(sendDict[rank][0])
    #        if index >= len(sendDict[rank]):
    #            Ln = len(sendDict[rank])
    #            Ln *= 6
    #            temp = np.ones( Ln )
    #            temp[:,0:index] = sendDict[rank]
    #            sendDict[rank] = temp
    #        sendDict[rank][index:index+6] = migBuff[i,:,:].flatten()
    #        sendDict[rank][0] = index+6
    #    else:
    #        sendDict[rank] = np.ones( buffLen + 1 )
    #        sendDict[rank][1:7] = migBuff[i,:,:].flatten()
    #        sendDict[rank][0]=7

    ##send migrated particles.
    #send_requests=[]
    #for i in mpiComm.neigs.ranks:
    #    if i in sendDict:
    #        Sreq = mpiComm.comm.Isend([sendDict[i][1:int(sendDict[i][0])]\
    #                          ,MPI.DOUBLE], dest = i, tag = mpiComm.rank)
    #        send_requests.append(Sreq)
    #    else:
    #        Sreq = mpiComm.comm.Isend([emptyArray, MPI.DOUBLE], dest = i,  tag = mpiComm.rank)
    #        send_requests.append(Sreq)

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

    send_requests=[]
    for i in range(size):
        index = int(sendBuff[i,0])
        if index == 1 and i != mpiComm.rank:
            Sreq = mpiComm.comm.Isend([ np.array([-50.,-50.,-50.,-50.,-50.,-50.]), MPI.DOUBLE], dest = i, tag = mpiComm.rank)
            send_requests.append(Sreq)
        elif i != mpiComm.rank:
            buff = sendBuff[i,1:index]
            Sreq = mpiComm.comm.Isend([buff, MPI.DOUBLE], dest = i, tag = mpiComm.rank)
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

        ptcl = pos[i,:].flatten()

        if cz == Upper:
            U_buff = np.append(U_buff,ptcl+U_shift)
        if cz == Upper and cy == North:
            UN_buff = np.append(UN_buff,ptcl+UN_shift)
        if cz == Upper and cy == South:
            US_buff = np.append(US_buff,ptcl+US_shift)
        if cz == Upper and cx == East:
            UE_buff = np.append(UE_buff,ptcl+UE_shift)
        if cz == Upper and cx == West:
            UW_buff = np.append(UW_buff,ptcl+UW_shift)
        if cz == Upper and cy == North and cx == East:
            UNE_buff = np.append(UNE_buff,ptcl+UNE_shift)
        if cz == Upper and cy == North and cx == West:
            UNW_buff = np.append(UNW_buff,ptcl+UNW_shift)
        if cz == Upper and cy == South and cx == East:
            USE_buff = np.append(USE_buff,ptcl+USE_shift)
        if cz == Upper and cy == South and cx == West:
            USW_buff = np.append(USW_buff,ptcl+USW_shift)

        if cy == North:
            N_buff = np.append(N_buff,ptcl+N_shift)
        if cy == South:
            S_buff = np.append(S_buff,ptcl+S_shift)
        if cx == East:
            E_buff = np.append(E_buff,ptcl+E_shift)
        if cx == West:
            W_buff = np.append(W_buff,ptcl+W_shift)
        if cy == North and cx == East:
            NE_buff = np.append(NE_buff,ptcl+NE_shift)
        if cy == North and cx == West:
            NW_buff = np.append(NW_buff,ptcl+NW_shift)
        if cy == South and cx == East:
            SE_buff = np.append(SE_buff,ptcl+SE_shift)
        if cy == South and cx == West:
            SW_buff = np.append(SW_buff,ptcl+SW_shift)

        if cz == Lower:
            D_buff = np.append(D_buff,ptcl+D_shift)
        if cz == Lower and cy == North:
            DN_buff = np.append(DN_buff,ptcl+DN_shift)
        if cz == Lower and cy == South:
            DS_buff = np.append(DS_buff,ptcl+DS_shift)
        if cz == Lower and cx == East:
            DE_buff = np.append(DE_buff,ptcl+DE_shift)
        if cz == Lower and cx == West:
            DW_buff = np.append(DW_buff,ptcl+DW_shift)
        if cz == Lower and cy == North and cx == East:
            DNE_buff = np.append(DNE_buff,ptcl+DNE_shift)
        if cz == Lower and cy == North and cx == West:
            DNW_buff = np.append(DNW_buff,ptcl+DNW_shift)
        if cz == Lower and cy == South and cx == East:
            DSE_buff = np.append(DSE_buff,ptcl+DSE_shift)
        if cz == Lower and cy == South and cx == West:
            DSW_buff = np.append(DSW_buff,ptcl+DSW_shift)

        ls[i] = head[c]
        head[c] = i


    #=== Particle Migration-Recv =============

    #probe incomming messages for total
    #number of received particles and
    #build recv buffers.
    recvB = []
    for i in mpiComm.neigs.ranks:
        probS = mpiComm.comm.Probe(status=s,source=i,tag=i)
        count = s.Get_count(datatype=MPI.DOUBLE)
        recvB.append(np.empty(count,dtype=np.float64))

    #recieve particles
    recv_requests=[]
    p = 0
    for i in mpiComm.neigs.ranks:
        Rreq = mpiComm.comm.Irecv([recvB[p], MPI.DOUBLE], source = i, tag = i)
        recv_requests.append(Rreq)
        p+=1
    recvPos = np.concatenate(recvB,axis=0)

    MPI.Request.waitall(send_requests)
    MPI.Request.waitall(recv_requests)

    #move migrated particles into rank's particle array
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
    for i in range(N,Nr):

        cx = int(np.floor((pos[i,0] - mpiComm.Lmin[0])/rc_x))
        cy = int(np.floor((pos[i,1] - mpiComm.Lmin[1])/rc_y))
        cz = int(np.floor((pos[i,2] - mpiComm.Lmin[2])/rc_z))
        c = cx + cy*Lxd + cz*Lxd*Lyd

        ptcl = pos[i,:].flatten()

        if cz == Upper:
            U_buff = np.append(U_buff,ptcl+U_shift)
        if cz == Upper and cy == North:
            UN_buff = np.append(UN_buff,ptcl+UN_shift)
        if cz == Upper and cy == South:
            US_buff = np.append(US_buff,ptcl+US_shift)
        if cz == Upper and cx == East:
            UE_buff = np.append(UE_buff,ptcl+UE_shift)
        if cz == Upper and cx == West:
            UW_buff = np.append(UW_buff,ptcl+UW_shift)
        if cz == Upper and cy == North and cx == East:
            UNE_buff = np.append(UNE_buff,ptcl+UNE_shift)
        if cz == Upper and cy == North and cx == West:
            UNW_buff = np.append(UNW_buff,ptcl+UNW_shift)
        if cz == Upper and cy == South and cx == East:
            USE_buff = np.append(USE_buff,ptcl+USE_shift)
        if cz == Upper and cy == South and cx == West:
            USW_buff = np.append(USW_buff,ptcl+USW_shift)

        if cy == North:
            N_buff = np.append(N_buff,ptcl+N_shift)
        if cy == South:
            S_buff = np.append(S_buff,ptcl+S_shift)
        if cx == East:
            E_buff = np.append(E_buff,ptcl+E_shift)
        if cx == West:
            W_buff = np.append(W_buff,ptcl+W_shift)
        if cy == North and cx == East:
            NE_buff = np.append(NE_buff,ptcl+NE_shift)
        if cy == North and cx == West:
            NW_buff = np.append(NW_buff,ptcl+NW_shift)
        if cy == South and cx == East:
            SE_buff = np.append(SE_buff,ptcl+SE_shift)
        if cy == South and cx == West:
            SW_buff = np.append(SW_buff,ptcl+SW_shift)

        if cz == Lower:
            D_buff = np.append(D_buff,ptcl+D_shift)
        if cz == Lower and cy == North:
            DN_buff = np.append(DN_buff,ptcl+DN_shift)
        if cz == Lower and cy == South:
            DS_buff = np.append(DS_buff,ptcl+DS_shift)
        if cz == Lower and cx == East:
            DE_buff = np.append(DE_buff,ptcl+DE_shift)
        if cz == Lower and cx == West:
            DW_buff = np.append(DW_buff,ptcl+DW_shift)
        if cz == Lower and cy == North and cx == East:
            DNE_buff = np.append(DNE_buff,ptcl+DNE_shift)
        if cz == Lower and cy == North and cx == West:
            DNW_buff = np.append(DNW_buff,ptcl+DNW_shift)
        if cz == Lower and cy == South and cx == East:
            DSE_buff = np.append(DSE_buff,ptcl+DSE_shift)
        if cz == Lower and cy == South and cx == West:
            DSW_buff = np.append(DSW_buff,ptcl+DSW_shift)

        ls[i] = head[c]
        head[c] = i

    N = len(pos[:,0])

    #copy particles for LCL.
    send_buff = [(mpiComm.neigs.U_rank,U_buff),(mpiComm.neigs.UN_rank,UN_buff)\
                ,(mpiComm.neigs.US_rank,US_buff),(mpiComm.neigs.UE_rank,UE_buff)\
                ,(mpiComm.neigs.UW_rank,UW_buff),(mpiComm.neigs.UNE_rank,UNE_buff)\
                ,(mpiComm.neigs.UNW_rank,UNW_buff),(mpiComm.neigs.USE_rank,USE_buff)\
                ,(mpiComm.neigs.USW_rank,USW_buff),(mpiComm.neigs.N_rank,N_buff)\
                ,(mpiComm.neigs.S_rank,S_buff),(mpiComm.neigs.E_rank,E_buff)\
                ,(mpiComm.neigs.W_rank,W_buff),(mpiComm.neigs.NE_rank,NE_buff)\
                ,(mpiComm.neigs.NW_rank,NW_buff),(mpiComm.neigs.SE_rank,SE_buff)\
                ,(mpiComm.neigs.SW_rank,SW_buff),(mpiComm.neigs.D_rank,D_buff)\
                ,(mpiComm.neigs.DN_rank,DN_buff),(mpiComm.neigs.DS_rank,DS_buff)\
                ,(mpiComm.neigs.DE_rank,DE_buff),(mpiComm.neigs.DW_rank,DW_buff)\
                ,(mpiComm.neigs.DNE_rank,DNE_buff),(mpiComm.neigs.DNW_rank,DNW_buff)\
                ,(mpiComm.neigs.DSE_rank,DSE_buff),(mpiComm.neigs.DSW_rank,DSW_buff)]


    #MPI Send
    #send_dict = {}
    #for rank, buff in send_buff:
    #    if rank in send_dict:
    #        send_dict[rank] = np.append(send_dict[rank],buff)
    #    else:
    #        send_dict[rank] = buff


    buffLen = 300
    sendBuff = np.ones( (size,buffLen+1))
    for rank, buff in send_buff:
        tempBuff = sendBuff[rank,:]
        index = int(tempBuff[0])
        while index + len(buff) >= buffLen:
            buffLen *= 3
            temp = np.ones( (mpiComm.size,buffLen+1))
            temp[:,0:index] = sendBuff
            sendBuff = temp
            tempBuff = sendBuff[rank,:]
        buffSize=len(buff)
        tempBuff[index:index+buffSize] = buff
        tempBuff[0] = index+buffSize


    #BUG: particles not sent to self!
    #send_requests = []
    #for i in mpiComm.neigs.ranks:
    #    Sreq = mpiComm.comm.Isend([sendBuff[i],MPI.DOUBLE], dest = i, tag = mpiComm.rank*10)
    #    send_requests.append(Sreq)

    send_requests=[]
    for i in mpiComm.neigs.ranks:
        index = int(sendBuff[i,0])
        if index == 1:
            Sreq = mpiComm.comm.Isend([ np.array([-50.,-50.,-50.,-50.,-50.,-50.]), MPI.DOUBLE], dest = i, tag = mpiComm.rank*10)
            send_requests.append(Sreq)
        else:
            buff = sendBuff[i,1:index]
            Sreq = mpiComm.comm.Isend([buff, MPI.DOUBLE], dest = i, tag = mpiComm.rank*10)
            send_requests.append(Sreq)


    #interior update
    for cx in range(1,Lxd-1):
        for cy in range(1,Lyd-1):
            for cz in range(1,Lzd-1):

                c = cx + cy*Lxd + cz*Lxd*Lyd

                for cz_N in range(cz-1,cz+2):
                    for cy_N in range(cy-1,cy+2):
                        for cx_N in range(cx-1,cx+2):

                            c_N = cx_N + cy_N*Lxd + cz_N*Lxd*Lyd

                            i = head[c]

                            while(i != empty):

                                j = head[c_N]

                                while(j != empty):

                                    if i < j:
                                        dx = pos[i,0] - pos[j,0]
                                        dy = pos[i,1] - pos[j,1]
                                        dz = pos[i,2] - pos[j,2]
                                        r = np.sqrt(dx**2 + dy**2 + dz**2)

                                        if r < rc:
                                            U_s_r = U_s_r + np.exp(-kappa*r)/r
                                            f1 = 1./r**2*np.exp(-kappa*r)
                                            f2 = kappa/r*np.exp(-kappa*r)
                                            fr = f1+f2
                                            acc_s_r[i,0] = acc_s_r[i,0] + fr*dx/r
                                            acc_s_r[i,1] = acc_s_r[i,1] + fr*dy/r
                                            acc_s_r[i,2] = acc_s_r[i,2] + fr*dz/r

                                            acc_s_r[j,0] = acc_s_r[j,0] - fr*dx/r
                                            acc_s_r[j,1] = acc_s_r[j,1] - fr*dy/r
                                            acc_s_r[j,2] = acc_s_r[j,2] - fr*dz/r

                                    j = ls[j]

                                i = ls[i]

    #probe incomming messages for total
    #number of received particles and
    #build recv buffers.
    recvB = []
    for i in mpiComm.neigs.ranks:
        probS = mpiComm.comm.Probe(status=s,source=i,tag=i*10)
        count = s.Get_count(datatype=MPI.DOUBLE)
        recvB.append(np.empty(count,dtype=np.float64))

    #recieve particles
    recv_requests=[]
    p = 0
    for i in mpiComm.neigs.ranks:
        Rreq = mpiComm.comm.Irecv([recvB[p], MPI.DOUBLE], source = i, tag = i*10)
        recv_requests.append(Rreq)
        p+=1

    MPI.Request.waitall(send_requests)
    MPI.Request.waitall(recv_requests)

    recvPos = np.concatenate(recvB,axis=0)

    #filter empty messages
    recvBuff = recvPos.reshape(int(len(recvPos)/3),3)
    recvBuff = recvBuff[np.where(recvBuff[:,0]!=-50.)]

    index = int(sendBuff[mpiComm.rank,0])
    if index == 1:
        selfBuff = np.array([],dtype=np.float64)
    else:
        selfBuff = sendBuff[mpiComm.rank,1:index]
        selfBuff = selfBuff.reshape(int(len(selfBuff)/3),3)
        selfBuff = selfBuff[np.where(selfBuff[:,0]!=-50.)]

    #add Copied Particles to particle position array
    pos = np.append(pos,recvBuff,axis=0)
    pos = np.append(pos,selfBuff,axis=0)

    #construct new linked cell list using
    # old LCL and copied particles
    Nb = len(pos[:,0])
    lsRecv  = np.arange(Nb)
    lsRecv[:N] = ls
    headRecv= np.empty(shape=((Lzd+2),(Lyd+2),(Lxd+2)),dtype=np.int32)
    headRecv.fill(empty)
    headRecv[1:Lzd+1,1:Lyd+1,1:Lxd+1] = head.reshape((Lzd,Lyd,Lxd))
    headRecv = headRecv.flatten()
    for i in range(N,Nb):
        cx = int(np.floor((rc_x + pos[i,0] - mpiComm.Lmin[0])/rc_x))
        cy = int(np.floor((rc_y + pos[i,1] - mpiComm.Lmin[1])/rc_y))
        cz = int(np.floor((rc_z + pos[i,2] - mpiComm.Lmin[2])/rc_z))
        c = cx + cy*(Lxd+2) + cz*(Lxd+2)*(Lyd+2)

        lsRecv[i] = headRecv[c]
        headRecv[c] = i

    #exterior update
    for I in exterior:

        cx = I[2]
        cy = I[1]
        cz = I[0]

        c = cx + cy*(Lxd+2) + cz*(Lxd+2)*(Lyd+2)

        for cz_N in range(cz-1,cz+2):
            for cy_N in range(cy-1,cy+2):
                for cx_N in range(cx-1,cx+2):

                    c_N = cx_N + cy_N*(Lxd+2) + cz_N*(Lxd+2)*(Lyd+2)

                    i = headRecv[c]

                    while(i != empty):

                        j = headRecv[c_N]

                        while(j != empty):

                            if i < j:

                                dx = pos[i,0] - pos[j,0]
                                dy = pos[i,1] - pos[j,1]
                                dz = pos[i,2] - pos[j,2]
                                r = np.sqrt(dx**2 + dy**2 + dz**2)

                                if r < rc:

                                    U_s_r = U_s_r + np.exp(-kappa*r)/r
                                    f1 = 1./r**2*np.exp(-kappa*r)
                                    f2 = kappa/r*np.exp(-kappa*r)
                                    fr = f1+f2
                                    acc_s_r[i,0] = acc_s_r[i,0] + fr*dx/r
                                    acc_s_r[i,1] = acc_s_r[i,1] + fr*dy/r
                                    acc_s_r[i,2] = acc_s_r[i,2] + fr*dz/r
                                    if j < N:
                                        acc_s_r[j,0] = acc_s_r[j,0] - fr*dx/r
                                        acc_s_r[j,1] = acc_s_r[j,1] - fr*dy/r
                                        acc_s_r[j,2] = acc_s_r[j,2] - fr*dz/r

                            j = lsRecv[j]

                        i = lsRecv[i]

    return U_s_r, acc_s_r[:N], vel, pos[:N]
