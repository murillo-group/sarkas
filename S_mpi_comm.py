import numpy as np
import sys
import S_global_names as glb
import S_constants as const

DEBUG=False

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
        self.neigs = self._initNeigs()


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

    def _initNeigs(self):

        #enforce Periodic Boundaries
        N = self.glbIndex[0] - 1
        if N < 0:
            N = self.decomp[0] - 1

        S = self.glbIndex[0] + 1
        if S > self.decomp[0]-1:
            S = 0

        W = self.glbIndex[1] - 1
        if W < 0:
            W = self.decomp[1] - 1

        E = self.glbIndex[1] + 1
        if E > self.decomp[1]-1:
            E = 0

        U = self.glbIndex[2] - 1
        if U < 0:
            U = self.decomp[2] - 1

        D = self.glbIndex[2] + 1
        if D > self.decomp[2]-1:
            D = 0

        #Calculate Global Index of Neigbhors
        Urank = (self.glbIndex[0],self.glbIndex[1],U)
        UNrank = (N,self.glbIndex[1],U)
        USrank = (S,self.glbIndex[1],U)
        UErank = (self.glbIndex[0],E,U)
        UWrank = (self.glbIndex[0],W,U)
        UNErank = (N,E,U)
        UNWrank = (N,W,U)
        USErank = (S,E,U)
        USWrank = (S,W,U)
        upperRanks = [[UNWrank,UNrank,UNErank]\
                              ,[UWrank,Urank,UErank]\
                              ,[USWrank,USrank,USErank]]

        Nrank = (N,self.glbIndex[1],self.glbIndex[2])
        Srank = (S,self.glbIndex[1],self.glbIndex[2])
        Erank = (self.glbIndex[0],E,self.glbIndex[2])
        Wrank = (self.glbIndex[0],W,self.glbIndex[2])
        NErank = (N,E,self.glbIndex[2])
        NWrank = (N,W,self.glbIndex[2])
        SErank = (S,E,self.glbIndex[2])
        SWrank = (S,W,self.glbIndex[2])
        midRanks = [[NWrank,Nrank,NErank]\
                              ,[Wrank,self.glbIndex,Erank]\
                              ,[SWrank,Srank,SErank]]

        Drank = (self.glbIndex[0],self.glbIndex[1],D)
        DNrank = (N,self.glbIndex[1],D)
        DSrank = (S,self.glbIndex[1],D)
        DErank = (self.glbIndex[0],E,D)
        DWrank = (self.glbIndex[0],W,D)
        DNErank = (N,E,D)
        DNWrank = (N,W,D)
        DSErank = (S,E,D)
        DSWrank = (S,W,D)
        downRanks = [[DNWrank,DNrank,DNErank]\
                              ,[DWrank,Drank,DErank]\
                              ,[DSWrank,DSrank,DSErank]]

        rtn = np.array([upperRanks,midRanks,downRanks])
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


