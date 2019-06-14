import numpy as np
import sys
import S_global_names as glb
import S_constants as const

DEBUG=False

class neigs:
    def __init__(self,mpiComm):

        self._rank = mpiComm.rank

        #enforce Periodic Boundaries
        W = mpiComm.glbIndex[0] - 1
        if W < 0:
            W = mpiComm.decomp[0] - 1

        E = mpiComm.glbIndex[0] + 1
        if E > mpiComm.decomp[0]-1:
            E = 0

        N = mpiComm.glbIndex[1] - 1
        if N < 0:
            N = mpiComm.decomp[1] - 1

        S = mpiComm.glbIndex[1] + 1
        if S > mpiComm.decomp[1]-1:
            S = 0

        U = mpiComm.glbIndex[2] - 1
        if U < 0:
            U = mpiComm.decomp[2] - 1

        D = mpiComm.glbIndex[2] + 1
        if D > mpiComm.decomp[2]-1:
            D = 0

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
        self.Nl = self._localPartNum()
        self.decomp = self._domain_decomp(self.size)
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

    def _localPartNum(self):
        Nlocal = int(glb.N/self.size)
        if self.rank < (glb.N%self.size):
            Nlocal += 1
        return Nlocal

    def _prime_factors(self,n):
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

    def _domain_decomp(self,n):
        pf= self._prime_factors(n)
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


