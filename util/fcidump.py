#!/usr/bin/python3
import ctf
import numpy as np

def write2Fcidump(integrals, kinetic, no, ms2=0, orbsym=1, isym=1, dtype='r'):
    '''
    Input a integral file in numpy ndarray format
    np,np,np,np
    '''
    nP = len(integrals[:,0,0,0])
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

    for i in range(nP):
        for a in range(nP):
            for j in range(nP):
                for b in range(nP):
                    if np.abs(integrals[i,a,j,b]) > 1e-8:
                        f.write("  " + str(integrals[i,a,j,b]) + "  " + str(i+1) \
                                + "  " + str(j+1) + "  " + str(a+1) + "  " + str(b+1) + "\n")
    for i in range(nP):
        f.write("  " + str(kinetic[i]) + "  " + str(i+1) + "  "\
                + str(i+1) +"  0  0\n")


    #for i in range(nP-no):
    #    f.write("  " + str(particleEnergies[i]) + "  " + str(i+no+1)\
    #            + "  0  0  0\n")

    f.write("  0.0  0  0  0  0")

    f.close()
    return
