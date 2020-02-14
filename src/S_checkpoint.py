'''
S_checkpoint.py

handling restarting/checkpoint
'''
import os
import numpy as np
import pickle


class Checkpoint:
    def __init__(self, params):
        self.params = params
        self.checkpoint_dir = self.params.Control.checkpoint_dir
        if not (os.path.exists(self.checkpoint_dir)):
            os.mkdir(self.checkpoint_dir)

        filename = self.checkpoint_dir + "/" + "S_parameters.pickle"

        save_file = open(filename, "wb")
        print('opened')
        pickle.dump(self.params, save_file)
        save_file.close()

    def dump(self, ptcls, it):
        species_id = ptcls.species_id
        species_name = ptcls.species_name
        pos = ptcls.pos
        vel = ptcls.vel
        acc = ptcls.acc
        file_name = self.checkpoint_dir+"/"+"S_checkpoint_"+str(it)
        np.savez(file_name, species_id=species_id, species_name=species_name, pos=pos, vel=vel, acc=acc)
