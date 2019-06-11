import numpy as np
import sys
import S_global_names as glb
import S_constants as const

DEBUG=False

def mpi_comm:
    def __init__(self,comm):
        self.comm = comm
        self.size = comm.size
        self.rank = comm.rank
        self.Nl = _localPartNum()
        self.decomp = _domain_decomp(self.size)
        self.Ll = np.array([glb.Lx/self.decomp[0]\
                           ,glb.Ly/self.decomp[1]\
                            glb.Lz/self.decomp[2]])

        LxMin = (self.rank%self.decomp[0])*self.Ll[0]
        LxMax = self.Ll[0] + LxMin

        LyMin = np.floor(self.rank/self.decomp[1])\
                        *self.Ll[1]
        LyMax = self.Ll[1]  + Lylocal

        LzMin = np.floor(self.rank/(self.decomp[0]\
                        *self.decomp[1]))*self.Ll[2]
        LzMax = self.Ll[2]  + Lzlocal

        self.Lmax = np.array([LxMax,LyMax,LzMax])
        self.Lmin = np.array([LxMin,LyMin,LzMin])
        self.glbIndex = glbIndexToRank(self.rank)
        self.neigs = _initNeigs()


    def _localPartNum(self):
        Nlocal = int(glb.N/self.size)
        if self.rank < (glb.N%self.size):
            Nlocal += 1
        return Nlocal

    def _prime_factors(n):
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

    def _domain_decomp(n):
        pf= _prime_factors(n)
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
        N = GlobalIndex[0] - 1
        if N < 0:
            N = decomp[0] - 1

        S = GlobalIndex[0] + 1
        if S > decomp[0]-1:
            S = 0

        W = GlobalIndex[1] - 1
        if W < 0:
            W = decomp[1] - 1

        E = GlobalIndex[1] + 1
        if E > decomp[1]-1:
            E = 0

        U = GlobalIndex[2] - 1
        if U < 0:
            U = decomp[2] - 1

        D = GlobalIndex[2] + 1
        if D > decomp[2]-1:
            D = 0

        #Calculate Global Index of Neigbhors
        Urank = (GlobalIndex[0],GlobalIndex[1],U)
        UNrank = (N,GlobalIndex[1],U)
        USrank = (S,GlobalIndex[1],U)
        UErank = (GlobalIndex[0],E,U)
        UWrank = (GlobalIndex[0],W,U)
        UNErank = (N,E,U)
        UNWrank = (N,W,U)
        USErank = (S,E,U)
        USWrank = (S,W,U)
        upperRanks = np.array([[UNWrank,UNrank,UNErank]\
                              ,[UWrank,Urank,UErank]\
                              ,[USWrank,USrank,USErank]])

        Nrank = (N,GlobalIndex[1],GlobalIndex[2])
        Srank = (S,GlobalIndex[1],GlobalIndex[2])
        Erank = (GlobalIndex[0],E,GlobalIndex[2])
        Wrank = (GlobalIndex[0],W,GlobalIndex[2])
        NErank = (N,E,GlobalIndex[2])
        NWrank = (N,W,GlobalIndex[2])
        SErank = (S,E,GlobalIndex[2])
        SWrank = (S,W,GlobalIndex[2])
        midRanks = np.array([[NWrank,Nrank,NErank]\
                              ,[Wrank,self.rank,Erank]\
                              ,[SWrank,Srank,SErank]])

        Drank = (GlobalIndex[0],GlobalIndex[1],D)
        DNrank = (N,GlobalIndex[1],D)
        DSrank = (S,GlobalIndex[1],D)
        DErank = (GlobalIndex[0],E,D)
        DWrank = (GlobalIndex[0],W,D)
        DNErank = (N,E,D)
        DNWrank = (N,W,D)
        DSErank = (S,E,D)
        DSWrank = (S,W,D)
        downRanks = np.array([[DNWrank,DNrank,DNErank]\
                              ,[DWrank,Drank,DErank]\
                              ,[DSWrank,DSrank,DSErank]])

        rtn = np.array([upperRanks,midRanks,downRanks])
    return rtn


    def rankToGlbIndex(rank):
        return (rank%decomp[0] \
               ,int(rank/decomp[0])%decomp[1]\
               ,int(rank/(decomp[0]*decomp[1])))

    def glbIndexToRank(glbIndex,decomp):
        return glbIndex[0] \
              +glbIndex[1]*decomp[0]\
              +glbIndex[2]*decomp[0]*decomp[1]


