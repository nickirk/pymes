import time
import numpy as np
import ctf
from ctf.core import *

def solve(tV_abij, tEpsilon_i, tEpsilon_a):
    '''
    mp2 algorithm
    '''
    algoName = "mp2.solve"
    timeMp2 = time.time()

    no = tEpsilon_i.size
    nv = tEpsilon_a.size

    print(algoName, f'no={no:d}, nv={nv:d}')
    
    tT_abij = ctf.tensor([nv,nv,no,no],dtype=complex,sp=1) 
    tT_abij = tV_abij
    tD_abij = ctf.tensor([nv,nv,no,no],dtype=complex, sp=1) 
    
    # the following ctf.einsum gives wrong result, with nan
    # the eisum gives the outer product of the input tensors
    #EinsumDabij = np.einsum('i,j,a,b->abij',tEpsilon_i.to_nparray(), tEpsilon_i.to_nparray(), (-tEpsilon_a).to_nparray(),(-tEpsilon_a).to_nparray())
    #tEinsumDabij = ctf.einsum('i,j,a,b->abij',tEpsilon_i, tEpsilon_i, -tEpsilon_a,-tEpsilon_a)
    #tEinsumDabij = ctf.astensor(EinsumDabij)
    
    # the following ctf expression calcs the outer sum, as wanted.
    tD_abij.i("abij") << tEpsilon_i.i("i") + tEpsilon_i.i("j")-tEpsilon_a.i("a")-tEpsilon_a.i("b")
    #tD_abij = ctf.tensor([no,no,nv,nv],dtype=complex, sp=1) 
    tD_abij = 1./tD_abij
    # why the ctf contraction is not used here?
    # let's see if the ctf contraction does the same job
    tT_abij = ctf.einsum('abij,abij->abij', tT_abij, tD_abij)
    
    # the following expression evaluate the sum of the two tensors on the right
    # to form an intermediate tensor
    # and then assign the sum of the intermediate tensor and the previous two tensors
    # on the same indices to the tensor on the left. Tensor contraction always
    # sum on the same indices!!
    # tT2_abij.i("abij") << tT2_abij.i('abij') +  tD_abij.i("abij")
    
    #write2Cc4sTensor(tT_abij.to_nparray(),[4,nv,nv,no,no],"Doubles")
    #edir = ctf.tensor([1],dtype=complex)
    #exc = ctf.tensor([1],dtype=complex)
    
    #edir.i("") << 2*tT_abij.i("abij") * tV_abij.i("abij")
    eDir  = 2.0*ctf.einsum('abij,abij->',tT_abij, tV_abij)
    
    # There is a bug with this contraction involving 
    # exchange integals, using einsum instead
    # tested against cc4s.
    #exc.i("") << -1.*tT_abij.i("abij") * tV_abij.i("baij")
    eExc = -1.0*ctf.einsum('abij,baij->',tT_abij, tV_abij)
    
    print("\tDirect contribution =",np.real(eDir))
    print("\tExchange contribution =", np.real(eExc))
    print("\tMP2 energy =",np.real(eDir+eExc))
    print("\t%f.3 seconds spent on Mp2" % (time.time()-timeMp2))
    return [eDir+eExc, tT_abij]
