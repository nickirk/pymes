import time
import numpy as np
import ctf
from ctf.core import *
from pymes.solver import mp2

def solve(tGamma_pqG, tEpsilon_i, tEpsilon_a, fDcd=False):
    '''
    ccd algorithm
    tV_ijkl = V^{ij}_{kl}
    tV_abij = V^{ab}_{ij}
    tT_abij = T^{ab}_{ij}
    the upper indices refer to conjugation
    '''
    algoName = "ccd.solve"
    timeCcd = time.time()

    no = tEpsilon_i.size
    nv = tEpsilon_a.size
    
    # parameters
    levelShift = 0.
    maxIter = 1000
    epsilonE = 1e-10
    delta = 1

    # construct the needed integrals here on spot.

    tV_abij = ctf.einsum("iaG, bjG -> abij", ctf.conj(tGamma_pqG[:no,no:,:]), tGamma_pqG[no:,:no,:])
    tV_ijab = ctf.einsum("aiG, jbG -> ijab", ctf.conj(tGamma_pqG[no:,:no,:]), tGamma_pqG[:no,no:,:])
    tV_klij = ctf.einsum("ikG, ljG -> klij", ctf.conj(tGamma_pqG[:no,:no,:]), tGamma_pqG[:no,:no,:])
    tV_iajb = ctf.einsum("jiG, abG -> iajb", ctf.conj(tGamma_pqG[:no,:no,:]), tGamma_pqG[no:,no:,:])
    tV_iabj = ctf.einsum("biG, ajG -> iabj", ctf.conj(tGamma_pqG[no:,:no,:]), tGamma_pqG[no:,:no,:])
    tV_abcd = ctf.einsum("caG, bdG -> abcd", ctf.conj(tGamma_pqG[no:,no:,:]), tGamma_pqG[no:,no:,:])
    
    eMp2, tT_abij = mp2.solve(tV_abij,tEpsilon_i,tEpsilon_a)

    tD_abij = ctf.tensor([nv,nv,no,no],dtype=complex, sp=1) 
    # the following ctf expression calcs the outer sum, as wanted.
    tD_abij.i("abij") << tEpsilon_i.i("i") + tEpsilon_i.i("j")\
            -tEpsilon_a.i("a")-tEpsilon_a.i("b")
    #tD_abij = ctf.tensor([no,no,nv,nv],dtype=complex, sp=1) 
    tD_abij = 1./(tD_abij+levelShift)
    # why the ctf contraction is not used here?
    # let's see if the ctf contraction does the same job
    dE = np.abs(np.real(eMp2))
    iteration = 0
    eLastIterCcd = np.real(eMp2)
    eCcd = 0.
    eDirCcd = 0.
    eExCcd = 0.
    print(algoName, ": Solving doubles amplitude equation")
    while np.abs(dE) > epsilonE and iteration < maxIter:
        iteration += 1
        tR_abij = getResidual(tEpsilon_i, tEpsilon_a, tT_abij, tV_klij, tV_ijab,\
                tV_abij, tV_iajb, tV_iabj, tV_abcd, fDcd)
        tT_abij += delta * ctf.einsum('abij,abij->abij', tR_abij, tD_abij)
        # update energy and norm of amplitudes
        eDirCcd, eExCcd = getEnergy(tV_abij, tT_abij)
        eCcd = np.real(eDirCcd + eExCcd)
        dE = eCcd - eLastIterCcd
        eLastIterCcd = eCcd

        l1NormT2 = ctf.norm(tT_abij)

        if iteration <= maxIter:
            print("\tIteration =", iteration)
            print("\t\tCorrelation Energy =", eCcd)
            print("\t\tL1 Norm of T2 =", l1NormT2)
        else:
            print("A converged solution is not found!")


    
    print("\tDirect contribution =",np.real(eDirCcd))
    print("\tExchange contribution =", np.real(eExCcd))
    print("\tCCD correlation energy =",eCcd)
    print("\t%f.3 seconds spent on CCD" % (time.time()-timeCcd))

    return [eCcd, tT_abij]

def getResidual(tEpsilon_i, tEpsilon_a, tT_abij, tV_klij, tV_ijab, tV_abij, tV_iajb, \
        tV_iabj, tV_abcd, fDcd):

    no = tEpsilon_i.size
    nv = tEpsilon_a.size
    tR_abij = ctf.tensor([nv,nv,no,no],dtype=complex,sp=1) 

    # tV_ijkl and tV_klij are not the same in transcorrelated Hamiltonian!
    tI_klij = tV_klij
    if not fDcd:
        tI_klij += ctf.einsum("klcd, cdij -> klij", tV_ijab, tT_abij)
    tR_abij.i("abij") << tV_abij.i("abij") + tV_abcd.i("abcd") * tT_abij.i("cdij")\
    + tI_klij.i("klij") * tT_abij.i("abkl") 



    # intermediates 
    # tTildeT_abij
    # tested using MP2 energy, the below tensor op is correct
    tTildeT_abij = ctf.tensor([nv,nv,no,no],dtype=complex,sp=1)
    tTildeT_abij.i("abij") << 2.0 * tT_abij.i("abij") - tT_abij.i("baij")

    # Xai_kbcj for the quadratic terms
    tXai_cbkj = ctf.einsum("klcd, dblj -> cbkj", tV_ijab, tTildeT_abij)

    tR_abij += ctf.einsum("acik, cbkj -> abij", tTildeT_abij, tXai_cbkj)

    # intermediate for exchange of ia and jb indices
    tFock_ab = ctf.tensor([nv,nv], dtype=complex, sp=1)
    tFock_ij = ctf.tensor([no,no], dtype=complex, sp=1)
    tFock_ab.i("aa") << tEpsilon_a.i("a")
    tFock_ij.i("ii") << tEpsilon_i.i("i")
    tX_ac = tFock_ab - 1./2.*ctf.einsum("adkl, lkdc -> ac", tTildeT_abij, tV_ijab)
    tX_ki = tFock_ij + 1./2.*ctf.einsum("cdil, lkdc -> ki", tTildeT_abij, tV_ijab)

    tEx_abij = ctf.tensor([nv,nv,no,no],dtype=complex,sp=1)
    tEx_baji = ctf.tensor([nv,nv,no,no],dtype=complex,sp=1)

    tEx_abij.i("abij") << tX_ac.i("ac") * tT_abij.i("cbij") \
                          - tX_ki.i("ki") * tT_abij.i("abkj") \
                          - tV_iajb.i("kaic") * tT_abij.i("cbkj")\
                          - tV_iajb.i("kbic") * tT_abij.i("ackj")\
                          + tTildeT_abij.i("acik") * tV_iabj.i("kbcj")

    tEx_baji.i("baji") << tEx_abij.i("baji") 
    #tEx_baji.i("baji") << tX_ac.i("bc") * tT_abij.i("caji") \
    #                        - tX_ki.i("kj") * tT_abij.i("baki") \
    #                        - tV_iajb.i("kbjc") * tT_abij.i("caki")\
    #                        - tV_iajb.i("kajc") * tT_abij.i("bcki")\
    #                        + tTildeT_abij.i("bcjk") * tV_iabj.i("kaci")
    # CCD has more terms than DCD
    if not fDcd:
        tXai_iabj = ctf.einsum("klcd, daki -> laci", tV_ijab, tT_abij)
        tEx_abij -= ctf.einsum("laci, cblj -> abij", tXai_iabj, tT_abij)
        tEx_abij += ctf.einsum("laci, bclj -> abij", tXai_iabj, tT_abij)

    ## !!!!!!! In TC method the following is not necessarily the same!!!!!!!!!!
    #tEx_baji.i("baji") << tEx_abij.i("abij")

    #tEx_abij.i("abij") << tEx_abij.i("abij") + tEx_abij.i("baji")
    #print(testEx_abij - tEx_abij)
    tR_abij += tEx_abij + tEx_baji
    #tR_abij += tEx_baji

    return tR_abij

def getEnergy(tV_abij, tT_abij):
    '''
    calculate the CCD energy, using the converged amplitudes
    '''
    tDirCcdE = 2. * ctf.einsum("abij, abij ->", tT_abij, tV_abij) 
    tExCcdE  = -1. * ctf.einsum("abij, baij ->", tT_abij, tV_abij)
    return [tDirCcdE, tExCcdE]
