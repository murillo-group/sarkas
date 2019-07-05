import numpy as np

data1 = np.load("Checkpoint/S_checkpoint_400.npz")
data2 = np.load("Checkpoint1/S_checkpoint_400.npz")

pos1 = data1["pos"]
vel1 = data1["vel"]
acc1 = data1["acc"]

pos2 = data2["pos"]
vel2 = data2["vel"]
acc2 = data2["acc"]

eps = 1.e-1000
print(np.allclose(pos1, pos2, rtol=eps))
print(np.allclose(vel1, vel2, rtol=eps))
print(np.allclose(acc1, acc2, rtol=eps))

print(np.array_equal(pos1, pos2))
print(np.array_equal(vel1, vel2))
print(np.array_equal(acc1, acc2))

