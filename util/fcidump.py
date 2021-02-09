#!/usr/bin/python3
import ctf
import numpy as np

def write2Fcidump(integrals, kinetic, no, ms2=1, orbsym=1, isym=1, dtype='r'):
    '''
    Input a integral file in ctf format
    np,np,np,np
    '''
    world = ctf.comm()

    nP = integrals.shape[0]
    inds, vals = integrals.read_all_nnz()
    if world.rank() == 0:
        f = open("FCIDUMP", "w")
        # write header
        f.write("&FCI\n")
        f.write(" NORB=%i,\n" % nP)
        f.write(" NELEC=%i,\n" % (no*2))
        f.write(" MS2=%i,\n" % ms2)
        #prepare orbsym
        OrbSym = [orbsym] * nP
        f.write(" ORBSYM="+str(OrbSym).strip('[]')+",\n")
        f.write(" ISYM=%i,\n" % isym)
        f.write("/\n")


        for l in range(len(inds)):
            p = int(inds[l]/nP**3)
            q = int((inds[l]-p*nP**3)/nP**2)
            r = int((inds[l]-p*nP**3-q*nP**2)/nP)
            s = int(inds[l]-p*nP**3-q*nP**2-r*nP)

            f.write("  " + str(vals[l]) + "  " + str(p+1) \
                    + "  " + str(r+1) + "  " + str(q+1)+ "  " + str(s+1) + "\n")

        for i in range(nP):
            f.write("  " + str(kinetic[i]) + "  " + str(i+1) + "  "\
                    + str(i+1) +"  0  0\n")


        #for i in range(nP-no):
        #    f.write("  " + str(particleEnergies[i]) + "  " + str(i+no+1)\
        #            + "  0  0  0\n")

        f.write("  0.0  0  0  0  0")

        f.close()
    return
