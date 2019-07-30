import numpy as np
import sys
import S_global_names as glb
import S_constants as const

DEBUG=False

class neigs:
    def __init__(self,mpiComm):

        self._rank = mpiComm.rank

        No_shift = np.zeros(3,dtype=np.float64)
        N_shift = 0.
        S_shift = 0.
        E_shift = 0.
        W_shift = 0.
        U_shift = 0.
        D_shift = 0.

        #enforce Periodic Boundaries
        W = mpiComm.glbIndex[0] - 1
        if W < 0:
            W = mpiComm.decomp[0] - 1
            W_shift =  glb.Lx

        E = mpiComm.glbIndex[0] + 1
        if E > mpiComm.decomp[0]-1:
            E = 0
            E_shift =  -glb.Lx

        N = mpiComm.glbIndex[1] - 1
        if N < 0:
            N = mpiComm.decomp[1] - 1
            N_shift =  glb.Ly

        S = mpiComm.glbIndex[1] + 1
        if S > mpiComm.decomp[1]-1:
            S = 0
            S_shift =  -glb.Ly

        U = mpiComm.glbIndex[2] - 1
        if U < 0:
            U = mpiComm.decomp[2] - 1
            U_shift =  glb.Lz

        D = mpiComm.glbIndex[2] + 1
        if D > mpiComm.decomp[2]-1:
            D = 0
            D_shift =  -glb.Lz

        #Calculate Global Index of Neigbhors (global Index)
        self.U_glbI = (mpiComm.glbIndex[0],mpiComm.glbIndex[1],U)
        self.UN_glbI = (mpiComm.glbIndex[0],N,U)
        self.US_glbI = (mpiComm.glbIndex[0],S,U)
        self.UE_glbI = (E,mpiComm.glbIndex[1],U)
        self.UW_glbI = (W,mpiComm.glbIndex[1],U)
        self.UNE_glbI = (E,N,U)
        self.UNW_glbI = (W,N,U)
        self.USE_glbI = (E,S,U)
        self.USW_glbI = (W,S,U)

        self.N_glbI = (mpiComm.glbIndex[0],N,mpiComm.glbIndex[2])
        self.S_glbI = (mpiComm.glbIndex[0],S,mpiComm.glbIndex[2])
        self.E_glbI = (E,mpiComm.glbIndex[1],mpiComm.glbIndex[2])
        self.W_glbI = (W,mpiComm.glbIndex[1],mpiComm.glbIndex[2])
        self.NE_glbI = (E,N,mpiComm.glbIndex[2])
        self.NW_glbI = (W,N,mpiComm.glbIndex[2])
        self.SE_glbI = (E,S,mpiComm.glbIndex[2])
        self.SW_glbI = (W,S,mpiComm.glbIndex[2])

        self.D_glbI = (mpiComm.glbIndex[0],mpiComm.glbIndex[1],D)
        self.DN_glbI = (mpiComm.glbIndex[0],N,D)
        self.DS_glbI = (mpiComm.glbIndex[0],S,D)
        self.DE_glbI = (E,mpiComm.glbIndex[1],D)
        self.DW_glbI = (W,mpiComm.glbIndex[1],D)
        self.DNE_glbI = (E,N,D)
        self.DNW_glbI = (W,N,D)
        self.DSE_glbI = (E,S,D)
        self.DSW_glbI = (W,S,D)

        #Calc Rank of Neigs
        self.U_rank = mpiComm.glbIndexToRank(self.U_glbI)
        self.UN_rank = mpiComm.glbIndexToRank(self.UN_glbI)
        self.US_rank = mpiComm.glbIndexToRank(self.US_glbI)
        self.UE_rank = mpiComm.glbIndexToRank(self.UE_glbI)
        self.UW_rank = mpiComm.glbIndexToRank(self.UW_glbI)
        self.UNE_rank = mpiComm.glbIndexToRank(self.UNE_glbI)
        self.UNW_rank = mpiComm.glbIndexToRank(self.UNW_glbI)
        self.USE_rank = mpiComm.glbIndexToRank(self.USE_glbI)
        self.USW_rank = mpiComm.glbIndexToRank(self.USW_glbI)


        self.N_rank = mpiComm.glbIndexToRank(self.N_glbI)
        self.S_rank = mpiComm.glbIndexToRank(self.S_glbI)
        self.E_rank = mpiComm.glbIndexToRank(self.E_glbI)
        self.W_rank = mpiComm.glbIndexToRank(self.W_glbI)
        self.NE_rank = mpiComm.glbIndexToRank(self.NE_glbI)
        self.NW_rank = mpiComm.glbIndexToRank(self.NW_glbI)
        self.SE_rank = mpiComm.glbIndexToRank(self.SE_glbI)
        self.SW_rank = mpiComm.glbIndexToRank(self.SW_glbI)


        self.D_rank = mpiComm.glbIndexToRank(self.D_glbI)
        self.DN_rank = mpiComm.glbIndexToRank(self.DN_glbI)
        self.DS_rank = mpiComm.glbIndexToRank(self.DS_glbI)
        self.DE_rank = mpiComm.glbIndexToRank(self.DE_glbI)
        self.DW_rank = mpiComm.glbIndexToRank(self.DW_glbI)
        self.DNE_rank = mpiComm.glbIndexToRank(self.DNE_glbI)
        self.DNW_rank = mpiComm.glbIndexToRank(self.DNW_glbI)
        self.DSE_rank = mpiComm.glbIndexToRank(self.DSE_glbI)
        self.DSW_rank = mpiComm.glbIndexToRank(self.DSW_glbI)

        #Calc Shifts
        self.U_shift = np.array([No_shift[0],No_shift[1],U_shift])
        self.UN_shift = np.array([No_shift[0],N_shift,U_shift])
        self.US_shift = np.array([No_shift[0],S_shift,U_shift])
        self.UE_shift = np.array([E_shift,No_shift[1],U_shift])
        self.UW_shift = np.array([W_shift,No_shift[1],U_shift])
        self.UNE_shift = np.array([E_shift,N_shift,U_shift])
        self.UNW_shift = np.array([W_shift,N_shift,U_shift])
        self.USE_shift = np.array([E_shift,S_shift,U_shift])
        self.USW_shift = np.array([W_shift,S_shift,U_shift])

        self.N_shift = np.array([No_shift[0],N_shift,No_shift[2]])
        self.S_shift = np.array([No_shift[0],S_shift,No_shift[2]])
        self.E_shift = np.array([E_shift,No_shift[1],No_shift[2]])
        self.W_shift = np.array([W_shift,No_shift[1],No_shift[2]])
        self.NE_shift = np.array([E_shift,N_shift,No_shift[2]])
        self.NW_shift = np.array([W_shift,N_shift,No_shift[2]])
        self.SE_shift = np.array([E_shift,S_shift,No_shift[2]])
        self.SW_shift = np.array([W_shift,S_shift,No_shift[2]])

        self.D_shift = np.array([No_shift[0],No_shift[1],D_shift])
        self.DN_shift = np.array([No_shift[0],N_shift,D_shift])
        self.DS_shift = np.array([No_shift[0],S_shift,D_shift])
        self.DE_shift = np.array([E_shift,No_shift[1],D_shift])
        self.DW_shift = np.array([W_shift,No_shift[1],D_shift])
        self.DNE_shift = np.array([E_shift,N_shift,D_shift])
        self.DNW_shift = np.array([W_shift,N_shift,D_shift])
        self.DSE_shift = np.array([E_shift,S_shift,D_shift])
        self.DSW_shift = np.array([W_shift,S_shift,D_shift])


        self.ranks = np.delete(np.unique(np.array([\
                         self.UNE_rank,self.UN_rank,self.UNW_rank\
                        ,self.UE_rank ,self.U_rank ,self.UW_rank\
                        ,self.USE_rank,self.US_rank,self.USW_rank\
                        ,self.NE_rank ,self.N_rank ,self.NW_rank\
                        ,self.E_rank  ,self._rank  ,self.W_rank\
                        ,self.SE_rank ,self.S_rank ,self.SW_rank\
                        ,self.DNE_rank,self.DN_rank,self.DNW_rank\
                        ,self.DE_rank ,self.D_rank ,self.DW_rank\
                        ,self.DSE_rank,self.DS_rank,self.DSW_rank])),self._rank)




class mpi_comm:
    def __init__(self,comm):
        self.comm = comm
        self.size = comm.size
        self.rank = comm.rank
        self.Nl = self.__localPartNum()
        self.decomp = self.__domain_decomp(self.size)
        self.Ll = np.array([glb.Lx/self.decomp[0]\
                           ,glb.Ly/self.decomp[1]\
                           ,glb.Lz/self.decomp[2]])

        LxMin = (self.rank%self.decomp[0])*self.Ll[0]
        LxMax = self.Ll[0] + LxMin

        LyMin = np.floor(self.rank/self.decomp[0])\
                        *self.Ll[1]
        LyMax = self.Ll[1]  + LyMin

        LzMin = np.floor(self.rank/(self.decomp[0]\
                        *self.decomp[1]))*self.Ll[2]
        LzMax = self.Ll[2]  + LzMin

        self.Lmax = np.array([LxMax,LyMax,LzMax])
        self.Lmin = np.array([LxMin,LyMin,LzMin])
        self.glbIndex = self.rankToGlbIndex(self.rank)
        self.neigs = neigs(self)
        self.interiorCells, self.exteriorCells = self.__interiorExterior()

    def __localPartNum(self):
        Nlocal = int(glb.N/self.size)
        if self.rank < (glb.N%self.size):
            Nlocal += 1
        return Nlocal

    def __prime_factors(self,n):
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors

    def __domain_decomp(self,n):
        pf= self.__prime_factors(n)
        Ln = len(pf)
        rtn = np.array([1,1,1])
        if Ln == 1:
            rtn = np.array([pf[0],1,1])
        elif Ln == 2:
            rtn = np.array([pf[0],pf[1],1])
        else:
            a = pf.pop(-1)
            b = pf.pop(-1)
            c = pf.pop(-1)
            rtn = np.array([c,b,a])
            while pf:
                fact = pf.pop(-1)
                rtn[np.argmin(rtn)] *= fact
        return rtn

    def __interiorExterior(self):

        Lxd = int(np.floor(self.Ll[0]/glb.rc))
        Lyd = int(np.floor(self.Ll[1]/glb.rc))
        Lzd = int(np.floor(self.Ll[2]/glb.rc))

        ix = np.linspace(1,Lxd,Lxd,dtype=np.int)
        iy = np.linspace(1,Lyd,Lyd,dtype=np.int)
        iz = np.linspace(1,Lzd,Lzd,dtype=np.int)
        yy,zz,xx = np.meshgrid(iy,iz,ix)
        zyx = np.dstack((zz.flatten(),yy.flatten(),xx.flatten()))[0]# - np.ones(3,dtype=np.int)

        interior=np.array([],dtype=np.int)
        intL = 0
        exterior=np.array([],dtype=np.int)
        extL = 0
        for I in zyx:
            cx = I[2]
            cy = I[1]
            cz = I[0]
            if cx == 1 or cx == Lxd:
                extL +=1
                exterior = np.append(exterior,I)
            elif cy == 1 or cy == Lyd:
                extL +=1
                exterior = np.append(exterior,I)
            elif cz == 1 or cz == Lzd:
                extL +=1
                exterior = np.append(exterior,I)
            else:
                intL +=1
                interior = np.append(interior,I)

        exterior = exterior.reshape(extL,3)
        interior = interior.reshape(intL,3)
        return (interior, exterior)


    def rankToGlbIndex(self,rank):
        return (rank%self.decomp[0] \
               ,int(rank/self.decomp[0])%self.decomp[1]\
               ,int(rank/(self.decomp[0]*self.decomp[1])))

    def glbIndexToRank(self,glbIndex):
        return glbIndex[0] \
              +glbIndex[1]*self.decomp[0]\
              +glbIndex[2]*self.decomp[0]*self.decomp[1]

    def posToGlb(self,pos):
        return (int(pos[0]/self.Ll[0])\
               ,int(pos[1]/self.Ll[1])\
               ,int(pos[2]/self.Ll[2]))

    def posToRank(self,pos):
        glbIndex = self.posToGlb(pos)
        rank = self.glbIndexToRank(glbIndex)
        return rank


