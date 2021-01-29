#!/usr/bin/python3
import ctf
import numpy as np



def write2Tcdump(tV_opqrst, dtype='r'):
    world = ctf.comm()

    nOrb = tV_opqrst.shape[0]

    # (or|ps|qt)

    if world.rank() == 0:
        f = open("TCDUMP", "w")
        inds, vals = tV_opqrst.read_all_nnz()
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
