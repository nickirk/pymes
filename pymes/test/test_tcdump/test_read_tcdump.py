import ctf
from ctf.core import *
import numpy as np
from numpy.core.numeric import full
import itertools

from pymes.log import print_logging_info
from pymes.util import tcdump
from pymes.integral import contraction

def main(file_name):
    t_V_orpsqt = tcdump.read(file_name, sp=0)
    print(t_V_orpsqt.sym)

    no = 2
    t_V_ijklmn = ctf.tensor([no,no,no,no,no,no], sym=t_V_orpsqt.sym, sp=t_V_orpsqt.sp)
    print(t_V_ijklmn.sym)
    t_V_ijklmn.i("ijklmn") << t_V_orpsqt[:no,:no,:no,:no,:no,:no].i("ijklmn")
    print(t_V_ijklmn.sym)
    full_tensor = t_V_ijklmn
    print(full_tensor.sym)

    print_logging_info("Testing exchange of the 1st electron pair indices...")
    sym = ctf.tensor([no,no,no,no,no,no], sym=t_V_ijklmn.sym, sp=t_V_orpsqt.sp) 
    sym.i("ijklmn") << full_tensor.i("jiklmn")
    assert(np.array_equal(full_tensor.to_nparray(),sym.to_nparray()))
    print_logging_info("Past!")

    print_logging_info("Testing exchange of the 2nd electron pair indices...")
    sym.set_zero()
    sym.i("ijklmn") << full_tensor.i("ijlkmn")
    assert(np.array_equal(full_tensor.to_nparray(),sym.to_nparray()))
    print_logging_info("Past!")

    print_logging_info("Testing exchange of the 3rd electron pair indices...")
    
    sym.set_zero()
    sym.i("ijklmn") << full_tensor.i("ijklnm")
    assert(np.array_equal(full_tensor.to_nparray(),sym.to_nparray()))
    print_logging_info("Past!")

    print_logging_info("Testing exchange of the 1st and 2nd electron pairs of indices...")
    sym.set_zero()
    sym.i("ijklmn") << full_tensor.i("klijnm")
    assert(np.array_equal(full_tensor.to_nparray(),sym.to_nparray()))
    print_logging_info("Past!")

    print_logging_info("Testing exchange of the 1st and 2nd electron pairs of indices and a pair of indices...")
    sym.set_zero()
    sym.i("ijklmn") << full_tensor.i("lkijnm")
    assert(np.array_equal(full_tensor.to_nparray(),sym.to_nparray()))
    print_logging_info("Past!")

    # write out the TCDUMP to compare with the original one
    tcdump.write_2_tcdump(t_V_orpsqt, file_name="TCDUMP.w")


def gen_sym_indices(string_inds):
    full_sym_inds = [string_inds]
    # exchange of two indices
    i = 0
    while i < 3:
        num_ele = len(full_sym_inds)
        for ind in range(num_ele):
            tmp_inds = list(full_sym_inds[ind])
            tmp_inds[i], tmp_inds[i+3] = tmp_inds[i+3], tmp_inds[i]
            full_sym_inds.append(''.join(tmp_inds))
            print(full_sym_inds)
        i += 1
        
    # exchange of a pair of indices
    tmp_inds = []
    for i in full_sym_inds:
        i = list(i)
        for per_1, per_2 in zip(itertools.permutations(i[:3]), itertools.permutations(i[3:])):
            
            per = ''.join(per_1+per_2)
            tmp_inds.append(per)
    full_sym_inds = tmp_inds
    #full_sym_inds = set(full_sym_inds)
    #return list(full_sym_inds)
    return full_sym_inds



if __name__ == '__main__':
    main("TCDUMP")