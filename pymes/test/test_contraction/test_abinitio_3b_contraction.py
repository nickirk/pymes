import ctf
import numpy as np
from pymes.integral import contraction
from pymes.util import tcdump


def test_triples_dense_tensor(tcdump_file="TCDUMP"):
    t_L_orpsqt_d = tcdump.read(tcdump_file, sp=0)
    no = 2
    t_T_0 = contraction.get_triple_contraction(no, t_L_orpsqt_d)

    assert np.abs(t_T_0 - 0.021384710425165987) < 1.e-8

    #
    # t_S_pqrs = contraction.get_double_contraction(no, t_L_orpsqt_d)

# the sparse test fails due to a bug in ctf, disable it for now

#def test_triples_sparse_tensor_test(tcdump_file="TCDUMP"):
#    t_L_orpsqt_s = tcdump.read(tcdump_file, sp=1)
#    no = 2
#    t_T_0 = contraction.get_triple_contraction(no, t_L_orpsqt_s)
#    assert np.abs(t_T_0 - 0.021384710425165987) < 1.e-8

    # single and double contractions still needed

def test_singles_dense_tensor(tcdump_file="TCDUMP"):
    t_L_orpsqt_d = tcdump.read(tcdump_file, sp=0)
    no = 2
    t_D_pqrs = contraction.get_single_contraction(no, t_L_orpsqt_d)
    t_D_qpsr = ctf.tensor(t_D_pqrs.shape, sp=t_D_pqrs.sp)
    # test if t_D_pqrs has particle exchange symmetry, ie <pq|rs> = <qp|sr>
    t_D_qpsr.i("qpsr") << t_D_pqrs.i("pqrs")
    assert np.sum(np.abs(t_D_pqrs.to_nparray()-t_D_qpsr.to_nparray())) < 1e-8

