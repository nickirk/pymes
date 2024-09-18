import numpy as np
from numpy.core.numeric import full

from pymes.log import print_logging_info
from pymes.util import tcdump
from pymes.integral import contraction
# deprecated

print("Warning: test_read_write_tcdump.py is deprecated. ")
def test_read_write(file_name="TCDUMP.H2.tc"):
    t_V_orpsqt = tcdump.read(file_name, sp=0)
    print(t_V_orpsqt.sym)

    no = 1
    t_V_ijklmn = np.zeros([no,no,no,no,no,no])
    print(t_V_ijklmn.sym)
    t_V_ijklmn.i("ijklmn") << t_V_orpsqt[:no,:no,:no,:no,:no,:no].i("ijklmn")
    print(t_V_ijklmn.sym)
    full_tensor = t_V_ijklmn
    print(full_tensor.sym)

    print_logging_info("Testing exchange of the 1st electron pair indices...")
    sym = np.zeros([no,no,no,no,no,no], sym=t_V_ijklmn.sym, sp=t_V_orpsqt.sp) 
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
    sym.i("ijklmn") << full_tensor.i("klijmn")
    assert(np.array_equal(full_tensor.to_nparray(),sym.to_nparray()))
    print_logging_info("Past!")

    print_logging_info("Testing exchange of the 1st and 2nd electron pairs of indices and a pair of indices...")
    sym.set_zero()
    sym.i("ijklmn") << full_tensor.i("lkijnm")
    assert(np.array_equal(full_tensor.to_nparray(),sym.to_nparray()))
    print_logging_info("Past!")

    # write out the TCDUMP to compare with the original one
    tcdump.write(t_V_orpsqt, file_name="TCDUMP.w")

def test_read_write_hdf5(file='tcdump.h5'):
    test_read_write(file)
