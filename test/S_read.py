import numpy as np

def init0(pos,vel,f_input):

    N = len(pos[:,0])

    vx, vy, vz, x, y, z, ax, ay, az = np.genfromtxt(f_input, usecols=(0,1,2,3,4,5,6,7,8), unpack=True)

    x = x[:N]
    y = y[:N]
    z = z[:N]

    vx = vx[:N]
    vy = vy[:N]
    vz = vz[:N]
    
    ax = ax[:N]
    ay = ay[:N]
    az = az[:N]

    x = x.reshape((N,1))
    y = y.reshape((N,1))
    z = z.reshape((N,1))

    vx = vx.reshape((N,1))
    vy = vy.reshape((N,1))
    vz = vz.reshape((N,1))
    
    ax = ax.reshape((N,1))
    ay = ay.reshape((N,1))
    az = az.reshape((N,1))

    pos = np.hstack((x,y,z))
    vel = np.hstack((vx,vy,vz))
    acc = np.hstack((ax,ay,az))
    
    return pos, vel
    

def initL(pos,vel,f_input):

    N = len(pos[:,0])

    x, y, z, vx, vy, vz, ax, ay, az = np.loadtxt(f_input, usecols=(0,1,2,3,4,5,6,7,8), unpack=True)

    x = x[:N]
    y = y[:N]
    z = z[:N]

    vx = vx[:N]
    vy = vy[:N]
    vz = vz[:N]
    
    ax = ax[:N]
    ay = ay[:N]
    az = az[:N]

    x = x.reshape((N,1))
    y = y.reshape((N,1))
    z = z.reshape((N,1))

    vx = vx.reshape((N,1))
    vy = vy.reshape((N,1))
    vz = vz.reshape((N,1))
    
    ax = ax.reshape((N,1))
    ay = ay.reshape((N,1))
    az = az.reshape((N,1))

    pos = np.hstack((x,y,z))
    vel = np.hstack((vx,vy,vz))
    acc = np.hstack((ax,ay,az))
    
    return pos, vel
