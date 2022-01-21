import ctf
import numpy as np
import itertools

from pymes.log import print_logging_info



def write(t_V_orpsqt, file_name="TCDUMP", sym=True, type='r', sp=1):
    world = ctf.comm()

    nOrb = t_V_orpsqt.shape[0]

    # (or|ps|qt)

    if world.rank() == 0:
        f = open(file_name, "w")
        f.write(str(nOrb)+"\n")
        inds, vals = t_V_orpsqt.read_all_nnz()
        for l in range(len(inds)):
            o = int(inds[l]/nOrb**5)
            r = int((inds[l]-o*nOrb**5)/nOrb**4)
            p = int((inds[l]-o*nOrb**5-r*nOrb**4)/nOrb**3)
            s = int((inds[l]-o*nOrb**5-r*nOrb**4-p*nOrb**3)/nOrb**2)
            q = int((inds[l]-o*nOrb**5-r*nOrb**4-p*nOrb**3-s*nOrb**2)/nOrb)
            t = int(inds[l]-o*nOrb**5-r*nOrb**4-p*nOrb**3-s*nOrb**2-q*nOrb)
            if np.abs(vals[l]) > 1e-10:
                #if (o <= p <= q) and (unique_index(o,r) <= unique_index(p,s) <= unique_index(q,t)):
                f.write(str(-vals[l]/3.)+" "+str(o+1)+" "+\
                            str(p+1)+" "+str(q+1)+" "+str(r+1)+" "+str(s+1)+" "\
                            +str(t+1)+"\n")
        f.close()
    return

def read(file_name="TCDUMP", sym=True, sp=1):
    '''
    Parameters:
    -----------
    file_name: string
               path to the tcdump file name, default TCDUMP in txt format
    sym: bool, whether to use symmetric tensor
    sp: int, 1 for sparse and 0 for dense ctf tensor
    Returns:
    --------
    t_V_opqrst: ctf tensor, sparse
                to use the symmetric tensor functionality in ctf, the indices will be in
                chemists' notation.
    '''
    # need to tell if the file is in hdf5 format. 
    print_logging_info("Reading in TCDUMP", level=1)
    SY = ctf.SYM.SY
    NS = ctf.SYM.NS
    if file_name == "*.h5" or file_name == "*.hdf5":
        print_logging_info("Integral file in hdf5 format.", level=1)
        integrals, indices, nb = _read_from_hdf5_tcdump("file_name")
    else:
        print_logging_info("Assuming integral file in txt format.", level=1)
        integrals, indices, nb = _read_from_txt_tcdump(file_name,sym=sym)
    # sp=1 plus sym does  not work in ctf.
    t_V_orpsqt = ctf.tensor([nb,nb,nb,nb,nb,nb], sp=sp, sym=[SY,NS,SY,NS,SY,NS])
    t_V_orpsqt.write(indices,integrals)
    #if sp == 0:
    #    t_V_orpsqt = ctf.tensor(t_V_orpsqt.to_nparray(), sp=0)
    return t_V_orpsqt

def _read_from_txt_tcdump(file_name="TCDUMP", sym=True):
    integrals = []
    indices = []
    with open(file_name, 'r') as reader:
        nb = int(reader.readline().strip())
        while True:
            line = reader.readline()
            if not line:
                break
            # TCDUMP uses physicists' notation for indices
            integral, o, p, q, r, s, t = line.split()
            integral = -3.*float(integral)
            o = int(o)-1
            p = int(p)-1
            q = int(q)-1
            r = int(r)-1
            s = int(s)-1
            t = int(t)-1
            # to use the symmetrise in ctf tensor, we put two indices 
            # that are exchangable next to each other, resulting chemists' notation for indices

            # manually include the 6-fold symmetries due to exchange of 3 electrons
            indices_sym = []
            #ints_sum = []
            for per_1, per_2 in zip(itertools.permutations([o,p,q]), itertools.permutations([r,s,t])):
                index = per_1[0]*nb**5+per_2[0]*nb**4+per_1[1]*nb**3+per_2[1]*nb**2+per_1[2]*nb**1+per_2[2]
                indices_sym.append(index)
            # remove repeated indices
            indices_sym = list(set(indices_sym))
            ints_sym = [integral] * len(indices_sym)
            integrals = integrals + ints_sym
            indices = indices + indices_sym
    return integrals, indices, nb


def _read_from_hdf5_tcdump(file_name="TCDUMP.hdf5"):
    import h5py
    # if hdf5 file format is used, try to read in parallel.
    # the tensor t_V_opqrst is stored as a sparse ctf tensor
    integrals = []
    indices = []
    nb = 0
    return integrals, indices, nb

def unique_index(p,q):
    return int(min(p,q)+(max(p,q)-1)*max(p,q)/2)
