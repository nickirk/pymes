import time
import numpy as np
import ctf
from ctf.core import *
from pymes.solver import mp2
from pymes.mixer import diis

def solve(tEpsilon_i, tEpsilon_a, tV_pqrs, levelShift=0., sp=0,  maxIter=100, fDcd=False, fDiis=True):
    '''
    ccd algorithm
    tV_ijkl = V^{ij}_{kl}
    tV_abij = V^{ab}_{ij}
    tT_abij = T^{ab}_{ij}
    the upper indices refer to conjugation
    '''
    algoName = "ccd.solve"
    timeCcd = time.time()
    world = ctf.comm()

    no = tEpsilon_i.size
    nv = tEpsilon_a.size
    
    # parameters
    levelShift = levelShift
    maxIter = maxIter
    epsilonE = 1e-8
    delta = 1.

    # construct the needed integrals here on spot.

    #tConjGamma_iaG = ctf.einsum("aiG->iaG", ctf.conj(tGamma_pqG[no:,:no,:]))
    #tV_iabj = ctf.einsum("ibG, ajG -> iabj", tConjGamma_iaG, tGamma_pqG[no:,:no,:])
    #tConjGamma_aiG = ctf.einsum("iaG->aiG", ctf.conj(tGamma_pqG[:no,no:,:]))
    #tV_aijb = ctf.einsum("ajG, ibG -> aijb", tConjGamma_aiG, tGamma_pqG[:no,no:,:])
    #tV_ijab = ctf.einsum("iaG, jbG -> ijab", tConjGamma_iaG, tGamma_pqG[:no,no:,:])
    #tConjGamma_ji = ctf.einsum("ijG->jiG", ctf.conj(tGamma_pqG[:no,:no,:]))
    #tV_klij = ctf.einsum("kiG, ljG -> klij", tConjGamma_ji, tGamma_pqG[:no,:no,:])
    #tV_iajb = ctf.einsum("ijG, abG -> iajb", tConjGamma_ji, tGamma_pqG[no:,no:,:])
    #tV_abij = ctf.einsum("aiG, bjG -> abij", tConjGamma_aiG, tGamma_pqG[no:,:no,:])
    #tConjGamma_ba = ctf.einsum("abG->baG", ctf.conj(tGamma_pqG[no:,no:,:]))
    #tV_abcd = ctf.einsum("acG, bdG -> abcd", tConjGamma_ba, tGamma_pqG[no:,no:,:])

    tV_iabj = tV_pqrs[:no,no:,no:,:no]
    tV_aijb = tV_pqrs[no:,:no,:no,no:]
    tV_ijab = tV_pqrs[:no,:no,no:,no:]
    tV_klij = tV_pqrs[:no,:no,:no,:no]
    tV_iajb = tV_pqrs[:no,no:,:no,no:]
    tV_abij = tV_pqrs[no:,no:,:no,:no]
    tV_abcd = tV_pqrs[no:,no:,no:,no:]
    
    eMp2, tT_abij = mp2.solve(tEpsilon_i,tEpsilon_a, tV_pqrs, levelShift, sp=sp)

    tD_abij = ctf.tensor([nv,nv,no,no],dtype=tV_pqrs.dtype, sp=sp) 
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
    if world.rank() == 0:
        print(algoName)
        print("\tUsing DCD: " , fDcd)
        print("\tSolving doubles amplitude equation")
        print("\tUsing data type %s" % tV_pqrs.dtype)
        print("\tUsing DIIS mixer: ", fDiis)
    residules = []
    amps = []
    mixSize = 4
    #tR_abij = ctf.tensor([nv,nv,no,no], dtype=complex, sp=1)
    while np.abs(dE) > epsilonE and iteration <= maxIter:
        iteration += 1
        tR_abij = 1.0*getResidual(tEpsilon_i, tEpsilon_a, tT_abij, tV_klij, tV_ijab,\
                tV_abij, tV_iajb, tV_iabj, tV_abcd, fDcd)
        tDeltaT_abij = ctf.einsum('abij,abij->abij', tR_abij, tD_abij)
        tT_abij += delta * tDeltaT_abij
        if fDiis:
            if len(residules) == 4:
                residules.pop(0)
                amps.pop(0)
            residules.append(tDeltaT_abij.copy())
            amps.append(tT_abij.copy())
            tT_abij = diis.mix(residules,amps)
        # update energy and norm of amplitudes
        eDirCcd, eExCcd = getEnergy(tT_abij, tV_ijab)
        eCcd = np.real(eDirCcd + eExCcd)
        dE = eCcd - eLastIterCcd
        eLastIterCcd = eCcd

        l1NormT2 = ctf.norm(tT_abij)

        if iteration < maxIter:
            if world.rank() == 0:
                print("\tIteration =", iteration)
                print("\t\tCorrelation Energy =", eCcd)
                print("\t\tdE =", dE)
                print("\t\tL1 Norm of T2 =", l1NormT2)
        else:
            if world.rank() == 0:
                print("A converged solution is not found!")


    
    if world.rank() == 0:
        print("\tDirect contribution =",np.real(eDirCcd))
        print("\tExchange contribution =", np.real(eExCcd))
        print("\tCCD correlation energy =",eCcd)
        print("\t%f.3 seconds spent on CCD" % (time.time()-timeCcd))

    return [eCcd, tT_abij]

def getResidual(tEpsilon_i, tEpsilon_a, tT_abij, tV_klij, tV_ijab, tV_abij, tV_iajb, \
        tV_iabj, tV_abcd, fDcd):

    algoName = "ccd.getResidual"
    no = tEpsilon_i.size
    nv = tEpsilon_a.size
    tR_abij = ctf.tensor([nv,nv,no,no],dtype=tV_klij.dtype,sp=tT_abij.sp) 

    # tV_ijkl and tV_klij are not the same in transcorrelated Hamiltonian!
    tI_klij = ctf.tensor([no,no,no,no], dtype=tV_klij.dtype,sp=tT_abij.sp)

    # = operatore pass the reference instead of making a copy.
    # if we want a copy, we need to specify that.
    #tI_klij = ctf.tensor([nv,nv,no,no], dtype=tV_klij.dtype,sp=tV_klij.sp)
    tI_klij += tV_klij
    if not fDcd:
        tI_klij += ctf.einsum("klcd, cdij -> klij", tV_ijab, tT_abij)

    tR_abij.i("abij") << tV_abij.i("abij") + tV_abcd.i("abcd") * tT_abij.i("cdij")\
    + tI_klij.i("klij") * tT_abij.i("abkl") 

    if not fDcd:
        tX_alcj = ctf.einsum("klcd, adkj -> alcj", tV_ijab, tT_abij)
        tR_abij += ctf.einsum("alcj, cbil -> abij", tX_alcj, tT_abij)



    # intermediates 
    # tTildeT_abij
    # tested using MP2 energy, the below tensor op is correct
    tTildeT_abij = ctf.tensor([nv,nv,no,no],dtype=tT_abij.dtype,sp=tT_abij.sp)
    tTildeT_abij.set_zero()
    tTildeT_abij.i("abij") << 2.0 * tT_abij.i("abij") - tT_abij.i("baij")

    # Xai_kbcj for the quadratic terms
    tXai_cbkj = ctf.einsum("klcd, dblj -> cbkj", tV_ijab, tTildeT_abij)

    tR_abij += ctf.einsum("acik, cbkj -> abij", tTildeT_abij, tXai_cbkj)

    # intermediate for exchange of ia and jb indices
    tFock_ab = ctf.tensor([nv,nv], dtype=tEpsilon_a.dtype, sp=0)
    tFock_ab.set_zero()
    tFock_ij = ctf.tensor([no,no], dtype=tEpsilon_i.dtype, sp=0)
    tFock_ij.set_zero()
    tFock_ab.i("aa") << tEpsilon_a.i("a")
    tFock_ij.i("ii") << tEpsilon_i.i("i")
    tX_ac = tFock_ab - 1./2 * ctf.einsum("adkl, lkdc -> ac", tTildeT_abij, tV_ijab)
    tX_ki = tFock_ij + 1./2 * ctf.einsum("cdil, lkdc -> ki", tTildeT_abij, tV_ijab)
    if not fDcd:
        tX_ac -= 1./2. * ctf.einsum("adkl, lkdc -> ac", tTildeT_abij, tV_ijab)
        tX_ki += 1./2. * ctf.einsum("cdil, lkdc -> ki", tTildeT_abij, tV_ijab)


    tEx_abij = ctf.tensor([nv,nv,no,no],dtype=tR_abij.dtype,sp=tR_abij.sp)
    tEx_baji = ctf.tensor([nv,nv,no,no],dtype=tR_abij.dtype,sp=tR_abij.sp)

    tEx_abij.i("abij") << tX_ac.i("ac") * tT_abij.i("cbij") \
                          - tX_ki.i("ki") * tT_abij.i("abkj") \
                          - tV_iajb.i("kaic") * tT_abij.i("cbkj")\
                          - tV_iajb.i("kbic") * tT_abij.i("ackj")\
                          + tTildeT_abij.i("acik") * tV_iabj.i("kbcj")
    if not fDcd:
        tXai_aibj = ctf.einsum("klcd, daki -> alci", tV_ijab, tT_abij)
        tEx_abij -= ctf.einsum("alci, cblj -> abij", tXai_aibj, tT_abij)
        tEx_abij += ctf.einsum("alci, bclj -> abij", tXai_aibj, tT_abij)

    tEx_baji.i("baji") << tEx_abij.i("abij") 
    #tEx_baji.i("baji") << tX_ac.i("bc") * tT_abij.i("caji") \
    #                        - tX_ki.i("kj") * tT_abij.i("baki") \
    #                        - tV_iajb.i("kbjc") * tT_abij.i("caki")\
    #                        - tV_iajb.i("kajc") * tT_abij.i("bcki")\
    #                        + tTildeT_abij.i("bcjk") * tV_iabj.i("kaci")
    # CCD has more terms than DCD

    ## !!!!!!! In TC method the following is not necessarily the same!!!!!!!!!!
    #tEx_baji.i("baji") << tEx_abij.i("abij")

    #tEx_abij.i("abij") << tEx_abij.i("abij") + tEx_abij.i("baji")
    #print(testEx_abij - tEx_abij)
    tR_abij += tEx_abij + tEx_baji
    #tR_abij += tEx_baji

    return tR_abij

def getEnergy(tT_abij, tV_ijab):
    '''
    calculate the CCD energy, using the converged amplitudes
    '''
    tDirCcdE = 2. * ctf.einsum("abij, ijab ->", tT_abij, tV_ijab) 
    tExCcdE  = -1. * ctf.einsum("abij, ijba ->", tT_abij, tV_ijab)
    return [tDirCcdE, tExCcdE]
