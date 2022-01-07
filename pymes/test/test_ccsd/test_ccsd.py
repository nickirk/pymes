import ctf
import numpy as np
from pymes.util import fcidump
from pymes.solver import ccsd, ccd, mp2
from pymes.mean_field import hf

def test_ccsd_energy(fcidump_file="FCIDUMP.LiH", ref_e=None):
    # known values
    hf_ref_e = -7.95197153899133
    n_elec, nb, e_core, e_orb, h_pq, V_pqrs = fcidump.read(fcidump_file)

    t_V_pqrs = ctf.astensor(V_pqrs)
    t_h_pq = ctf.astensor(h_pq)


    no = int(n_elec/2)
    # make sure HF energy is correct first
    hf_e = hf.calc_hf_e(no, e_core, t_h_pq, t_V_pqrs)
    print(hf_e)
    assert np.abs(hf_ref_e - hf_e) < 1.e-8

    # then MP2 and CCSD energies
    t_fock_pq = hf.construct_hf_matrix(no, t_h_pq, t_V_pqrs)
    mycc = ccsd.CCSD(no)
    ccsd_e = mycc.solve(t_fock_pq, t_V_pqrs)["ccsd e"]
    if ref_e is None:
        ref_e = -0.02035412567214456
    assert np.abs(ccsd_e - ref_e) < 1.e-7

def test_ccsd_fno(fcidump_file="fcidump.no"):
    test_ccsd_energy(fcidump_file, ref_e=-0.01931436971985408)
