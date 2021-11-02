import ctf
from ctf.core import *
import numpy as np

from pymes.logging import print_logging_info



def write_2_tcdump(t_V_opqrst, dtype='r'):
    world = ctf.comm()

    nOrb = t_V_opqrst.shape[0]

    # (or|ps|qt)

    if world.rank() == 0:
        f = open("TCDUMP", "w")
        inds, vals = t_V_opqrst.read_all_nnz()
        for l in range(len(inds)):
            o = int(inds[l]/nOrb**5)
            p = int((inds[l]-o*nOrb**5)/nOrb**4)
            q = int((inds[l]-o*nOrb**5-p*nOrb**4)/nOrb**3)
            r = int((inds[l]-o*nOrb**5-p*nOrb**4-q*nOrb**3)/nOrb**2)
            s = int((inds[l]-o*nOrb**5-p*nOrb**4-q*nOrb**3-r*nOrb**2)/nOrb)
            t = int(inds[l]-o*nOrb**5-p*nOrb**4-q*nOrb**3-r*nOrb**2-s*nOrb)
            if np.abs(vals[l]) > 1e-10:
            # depends on the correlator, sometimes there are 0 integrals when
            # there is a k=0. But in general, this is not the case
                f.write(str(vals[l])+" "+str(o+1)+" "+\
                        str(p+1)+" "+str(q+1)+" "+str(r+1)+" "+str(s+1)+" "\
                        +str(t+1)+"\n")
        f.close()
    return

def read_from_tcdump(file_name="TCDUMP"):
    '''
    Parameters:
    -----------
    file_name: string
               path to the tcdump file name, default TCDUMP in txt format
    Returns:
    --------
    t_V_opqrst: ctf tensor, sparse
    '''
    # need to tell if the file is in hdf5 format. 
    print_logging_info("Reading in TCDUMP", level=0)
    if file_name == "*.hdf5":
        print_logging_info("Integral file in hdf5 format.", level=1)
        integrals, indices, nb=__read_from_hdf5_tcdump("file_name")
    else:
        print_logging_info("Assuming integral file in txt format.", level=1)
        integrals, indices, nb = __read_from_txt_tcdump(file_name)
    t_V_opqrst = ctf.tensor([nb,nb,nb,nb,nb,nb], sp=1)
    t_V_opqrst.write(indices,integrals)
    return t_V_opqrst

def __read_from_txt_tcdump(file_name="TCDUMP"):
    integrals = []
    indices = []
    with open(file_name, 'r') as reader:
        nb = int(reader.readline().strip())
        while True:
            line = reader.readline()
            #if not line.strip():
            #    continue
            if not line:
                break
            integral, o, r, p, s, q, t = line.split()
            integral = float(integral)
            o = int(o)-1
            p = int(p)-1
            q = int(q)-1
            r = int(r)-1
            s = int(s)-1
            t = int(t)-1
            index = o*nb**5+p*nb**4+q*nb**3+r*nb**2+s*nb**1+t
            integrals.append(integral)
            indices.append(index)
    return integrals, indices, nb


def __read_from_hdf5_tcdump(file_name="TCDUMP.hdf5"):
    import h5py
    # if hdf5 file format is used, try to read in parallel.
    # the tensor t_V_opqrst is stored as a sparse ctf tensor
    integrals = []
    indices = []
    nb = 0
    return integrals, indices, nb
