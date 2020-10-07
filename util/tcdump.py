#!/usr/bin/python3
import ctf
import numpy as np



def write2Tcdump(integrals, ms2=0, orbsym=1, isym=1, dtype='r'):
    world = ctf.comm()

    nOrb = integrals.shape[0]

    print("shape of tensor=", nOrb)
    # (ia|jb|kc)
    if world.rank() == 0:
        f = open("TCDUMP", "w")
        inds, vals = integrals.read_all_nnz()
        for l in range(len(inds)):
            i = int(inds[l]/nOrb**5)
            a = int((inds[l]-i*nOrb**5)/nOrb**4)
            j = int((inds[l]-i*nOrb**5-a*nOrb**4)/nOrb**3)
            b = int((inds[l]-i*nOrb**5-a*nOrb**4-j*nOrb**3)/nOrb**2)
            k = int((inds[l]-i*nOrb**5-a*nOrb**4-j*nOrb**3-b*nOrb**2)/nOrb)
            c = int(inds[l]-i*nOrb**5-a*nOrb**4-j*nOrb**3-b*nOrb**2-k*nOrb)
            if np.abs(vals[l]) > 1e-10:
            # depends on the correlator, sometimes there are 0 integrals when
            # there is a k=0. But in general, this is not the case
                f.write(str(vals[l])+" "+str(i+1)+" "+\
                        str(j+1)+" "+str(k+1)+" "+str(a+1)+" "+str(b+1)+" "\
                        +str(c+1)+"\n")
        f.close()
    
    return
