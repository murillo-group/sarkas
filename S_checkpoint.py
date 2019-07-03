'''
S_checkpoint.py

handling restarting/checkpoint
'''
import os
import numpy as np
from inspect import currentframe, getframeinfo
import time
import pickle

class checkpoint:
    def __init__(self, params):
        self.params = params
        self.checkpoint_dir = "Checkpoint/"
        if not (os.path.exists(self.checkpoint_dir)):
            os.mkdir(self.checkpoint_dir)
        
        pickle.dump(self.params, open(self.checkpoint_dir+"S_parameters.pickle", "wb"))

    def dump(self, pos, vel, acc, it):
        file_name = self.checkpoint_dir+"S_checkpoint_"+str(it)
        np.savez(file_name, pos = pos, vel = vel, acc = acc)
