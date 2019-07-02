import numpy as np
data = np.load("Restart/restart_70.npz")
print(data.files)
pos = data['pos']
vel = data["vel"]
acc = data["acc"]

print(pos)
