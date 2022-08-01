import h5py

from pymes.log import print_logging_info

'''
This file contains functions to read in the tcfactors produced by TCHINT. 
Adapted from modules in NECI. License and Copyright info:
!  Copyright (c) 2013, Ali Alavi, the following procedure has been copied and modified from the NECI program which 
   is licenced under GNU GPL v.3 licence

Author: Ke Liao
'''

def read(file_name="tcfactors.h5"):
    if file_name == "*.h5" or file_name == "*.hdf5":
        print_logging_info("Reading tcfactors in hdf5 format...")
        tcfac_ = _read_h5(file_name)
    else:
        raise NameError("Reading txt format not implemented!")
    return tcfac_

def _read_h5(file_name):

    f = h5py.File(file_name, 'r')
    n_orb = f['nBasis']
    n_grid = f['nGrid']

    # loading weigths of the grids
    weights = f["weights"]
    assert(len(weights)==n_grid)

    # molecular orbital vals on grids
    mo_vals = f["mo_vals"]

    # ycoulomb
    ycoulomb = f["ycoulomb"]

    return n_orb, n_grid, weights, mo_vals, ycoulomb


