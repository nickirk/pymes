import numpy as np
from pymes.integral import contraction
from pymes.util import tcdump


def test_singles_dense_tensor(tcdump_file="TCDUMP"):
    t_L_orpsqt_d = tcdump.read(tcdump_file, sp=0)
    no = 2
    t_T_0 = contraction.get_triple_contraction(no, t_L_orpsqt_d)

    assert np.abs(t_T_0 - 1.859390082220713E-002) < 1.e-12

    #
    # t_S_pqrs = contraction.get_double_contraction(no, t_L_orpsqt_d)


def test_singles_sparse_tensor_test(tcdump_file="TCDUMP"):
    t_L_orpsqt_s = tcdump.read(tcdump_file, sp=1)
    no = 2
    t_T_0 = contraction.get_triple_contraction(no, t_L_orpsqt_s)
    assert np.abs(t_T_0 - 1.859390082220713E-002) < 1.e-12
    # single and double contractions still needed


if __name__ == '__main__':
    main(fcidump_file="FCIDUMP.be_test", tcdump_file="TCDUMP")
