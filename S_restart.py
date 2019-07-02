'''
S_restart.py

handling restarting/checkpoint
'''
import os
import numpy as np
from inspect import currentframe, getframeinfo
import time

class restart:
    def __init__(self):
        self.restart_dir = "Restart"
        if not (os.path.exists(self.restart_dir)):
            os.mkdir(self.restart_dir)

    def dump(self, pos, vel, acc, it):
        file_name = self.restart_dir+"/restart_"+str(it)
        np.savez(file_name, pos=pos, vel=vel, acc=acc) 
