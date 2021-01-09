import numpy as np
import numba as nb
import sys

@nb.jit
def calc_n(pos, qv, Nq, Nt, N, n_q_t):
    for it in range(Nt):
        print("it = ", it)
        for iqv in range(Nq):
            q_p = qv[iqv]
            n_q_t[it,iqv,0] = np.sum(np.exp(-1j*q_p*pos[it, :, 0]))
            n_q_t[it,iqv,1] = np.sum(np.exp(-1j*q_p*pos[it, :, 1]))
            n_q_t[it,iqv,2] = np.sum(np.exp(-1j*q_p*pos[it, :, 2]))

    return n_q_t

Nt = 5000
N = 10000
L = (4.0*np.pi*N/3.0)**(1.0/3.0)      # box length
dq = 2.*np.pi/L
print("L = ", L)
aw = 5.282005e-11
wp = 1.675504e15

q_max = 5
Nq = 3*int(q_max/dq)
print(Nt)
print(dq, Nq)
qv = np.zeros(Nq)

pos = np.zeros((Nt, N, 3))
pos = np.load("pos.npy")

if 0:
    dir1 = "OCP_mks/"
    for i in range(0, Nt):
        if(i%100 == 0):
            print(i)
        data = np.load(dir1+"Particles_Data/S_checkpoint_"+str(i*10)+".npz")
        pos[i] = data["pos"]/aw

for iqv in range(0, Nq, 3):
    iq = iqv/3.
    qv[iqv] = (iq+1.)*dq
    qv[iqv+1] = (iq+1.)*np.sqrt(2.)*dq
    qv[iqv+2] = (iq+1.)*np.sqrt(3.)*dq

n_qt = np.zeros((Nt, Nq, 3),dtype='complex128') #

print("nqt calculating....")
n_qt = calc_n(pos, qv, Nq, Nt, N, n_qt)

np.save("n_qt", n_qt)
