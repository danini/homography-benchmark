import h5py
from shutil import copyfile
import numpy as np
import os

def load_h5(filename):
    '''Loads dictionary from hdf5 file'''
    dict_to_load = {}
    with h5py.File(filename, 'r') as f:
        keys = [key for key in f.keys()]
        for key in keys:
            dict_to_load[key] = f[key][()]
    return dict_to_load

def copy_and_update_h5(input_filename, output_filename, data):
    '''Copy and update hdf5 file'''
    if os.path.exists(output_filename):
        os.remove(output_filename)
    copyfile(input_filename, output_filename)

    # Update file
    with h5py.File(output_filename, 'r+') as f:
        for key in f.keys():   
            f[key][()] = data[key]


def save_results(filename, data):
    f = open(filename, "a")
    for item in data:
        f.write(str(item) + ",")
    f.write("\n")
    f.close()