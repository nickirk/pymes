
import ctf
from ctf.core import *
import numpy as np

from pymes.logging import print_logging_info
from pymes.util import tcdump

def main(file_name):
    # def known values
    print_logging_info("Random entries test...")
    known_vals = [4.3547807609918094E-003,5.1685802980897883E-006, 1.5086553250634954E-005]
    known_inds = [[0,13,0,0,0,13],[1,11,2,11,1,7],[8,10,7,9,5,10]]
    t_V_opqrst = tcdump.read_from_tcdump(file_name)
    inds, vals = t_V_opqrst.read_all_nnz()
    nb = 14
    for index in range(len(known_vals)):
        print(known_inds[index])
        loc = 0
        for l in range(len(known_inds[index])):
            loc += known_inds[index][-l-1]*nb**l
        assert(np.abs(vals[np.where(inds == loc)]-known_vals[index]) < 1e-12)
    print_logging_info("Past!")

    # symmetries

if __name__ == '__main__':
    main("TCDUMP")