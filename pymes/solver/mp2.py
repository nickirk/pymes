import time
import numpy as np
import ctf

from pymes.log import print_logging_info

def solve(tEpsilon_i, tEpsilon_a, tV_pqrs, levelShift=0., sp=0):
    '''
    mp2 algorithm
    '''
    algoName = "mp2.solve"
    world = ctf.comm()
    timeMp2 = time.time()
    print_logging_info(algoName,level=0)

    no = tEpsilon_i.size
    nv = tEpsilon_a.size

    tV_ijab = tV_pqrs[:no,:no,no:,no:]
    tV_abij = tV_pqrs[no:,no:,:no,:no]


    tT_abij = ctf.tensor([nv,nv,no,no],dtype=tV_abij.dtype,sp=sp)
    #tT_abij += tV_abij
    tD_abij = ctf.tensor([nv,nv,no,no],dtype=tV_abij.dtype, sp=sp)

    # the following ctf.einsum gives wrong result, with nan
    # the eisum gives the outer product of the input tensors
    #EinsumDabij = np.einsum('i,j,a,b->abij',tEpsilon_i.to_nparray(), tEpsilon_i.to_nparray(), (-tEpsilon_a).to_nparray(),(-tEpsilon_a).to_nparray())
    #tEinsumDabij = ctf.einsum('i,j,a,b->abij',tEpsilon_i, tEpsilon_i, -tEpsilon_a,-tEpsilon_a)
    #tEinsumDabij = ctf.astensor(EinsumDabij)

    # the following ctf expression calcs the outer sum, as wanted.
    tD_abij.i("abij") << tEpsilon_i.i("i") + tEpsilon_i.i("j")-tEpsilon_a.i("a")-tEpsilon_a.i("b")
    #tD_abij = ctf.tensor([no,no,nv,nv],dtype=complex, sp=1)
    tD_abij = 1./(tD_abij+levelShift)
    # why the ctf contraction is not used here?
    # let's see if the ctf contraction does the same job
    tT_abij = ctf.einsum('abij,abij->abij', tV_abij, tD_abij)
    #tT_abij.i("abij") << tV_abij.i("abij") * tD_abij.i("abij")

    # the following expression evaluate the sum of the two tensors on the right
    # to form an intermediate tensor
    # and then assign the sum of the intermediate tensor and the previous two tensors
    # on the same indices to the tensor on the left. Tensor contraction always
    # sum on the same indices!!
    # tT2_abij.i("abij") << tT2_abij.i('abij') +  tD_abij.i("abij")

    #write2Cc4sTensor(tT_abij.to_nparray(),[4,nv,nv,no,no],"Doubles")

    eDir  = 2.0*ctf.einsum('abij,ijab->',tT_abij, tV_ijab)
    eExc = -1.0*ctf.einsum('abij,ijba->',tT_abij, tV_ijab)
    # There is a bug with this contraction involving
    # exchange integals, using einsum instead
    # tested against cc4s.
    #print_logging_info(edir)

    print_logging_info("Direct contribution = {:.12f}".format(np.real(eDir)),\
                       level=1)
    print_logging_info("Exchange contribution = {:.12f}".format(np.real(eExc)),\
                       level=1)
    print_logging_info("MP2 energy = {:.12f}".format(np.real(eDir+eExc)), level=1)
    print_logging_info("{:.3f} seconds spent on "\
                       .format((time.time()-timeMp2))+algoName, level=1)
    return [eDir+eExc, tT_abij]
