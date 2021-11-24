
import ctf
from ctf.core import *
import numpy as np
from numpy.core.numeric import full
import itertools

from pymes.logging import print_logging_info
from pymes.util import tcdump

def main(file_name):
    # def known values
    #print_logging_info("Random entries test...")
    #known_vals = [4.3547807609918094E-003,5.1685802980897883E-006, 1.5086553250634954E-005]
    #known_inds = [[0,0,0,13,0,13],[1,2,1,11,11,7],[8,7,5,10,9,10]]
    t_V_opqrst = tcdump.read_from_tcdump(file_name)
    #inds, vals = t_V_opqrst.read_all_nnz()
    #nb = 14
    #for index in range(len(known_vals)):
    #    print(known_inds[index])
    #    loc = 0
    #    for l in range(len(known_inds[index])):
    #        loc += known_inds[index][-l-1]*nb**l
    #    assert(np.abs(vals[np.where(inds == loc)]-known_vals[index]) < 1e-12)
    #print_logging_info("Past!")

    # symmetries
    #sym = ctf.einsum("opqrst -> rpqost", t_V_opqrst)
    #inds_sym, vals_sym = sym.read_all_nnz()
    #inds, vals = t_V_opqrst.read_all_nnz()
    #print(inds_sym, inds)
    #assert(np.array_equal(sym.to_nparray(),t_V_opqrst.to_nparray()))

    #sym = ctf.einsum("opqrst ->", sym)
    #print_logging_info("Exchange of 1. and 4. indices = ", sym)

    #sym = ctf.einsum("opqrst -> poqsrt", t_V_opqrst) - t_V_opqrst
    #sym = ctf.einsum("opqrst ->", sym)
    #print_logging_info("Exchange of 1. and 4. electrons = ", sym)
    full_inds = gen_sym_indices('opqrst')
    full_tensor = t_V_opqrst
    i = 1
    #for inds in full_inds[1:]:
    #    print(inds)
    #    print(i)
    #    i += 1
    #    full_tensor += ctf.einsum("opqrst -> "+inds, t_V_opqrst)
    
    # now remove repeated summations
    repeated_entries = gen_sym_indices("opqost")
    print(len(repeated_entries))

    sym = ctf.einsum("opqrst -> rpqost", full_tensor)
    assert(np.array_equal(full_tensor.to_nparray(),sym.to_nparray()))

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
        i += 1
        
    # exchange of a pair of indices
    tmp_inds = []
    for i in full_sym_inds:
        i = list(i)
        for per_1, per_2 in zip(itertools.permutations(i[:3]), itertools.permutations(i[3:])):
            
            per = ''.join(per_1+per_2)
            tmp_inds.append(per)
    full_sym_inds = tmp_inds
    full_sym_inds = set(full_sym_inds)
    return list(full_sym_inds)



if __name__ == '__main__':
    main("TCDUMP")