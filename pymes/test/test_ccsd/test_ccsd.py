import numpy as np
from pymes.util import fcidump
from pymes.solver import  ccd, ccsd
from pymes.mean_field import hf


def test_ccsd_energy(fcidump_file="./pymes/test/test_ccsd/FCIDUMP.LiH.321g", ref_e={"hf_e": -7.92958534362757, "ccsd_e": -0.01908832712812761,
                                                             "ccd_e": -0.01830250126018896}):
    n_elec, nb, e_core, e_orb, h_pq, V_pqrs = fcidump.read(fcidump_file)


    no = int(n_elec / 2)
    # make sure HF energy is correct first
    hf_e = hf.calc_hf_e(no, e_core, h_pq, V_pqrs)
    print(hf_e)
    assert np.isclose(hf_e, ref_e["hf_e"])

    # then MP2 and CCSD energies
    t_fock_pq = hf.construct_hf_matrix(no, h_pq, V_pqrs)
    myccd = ccd.CCD(no)
    ccd_e = myccd.solve(t_fock_pq, V_pqrs)["ccd e"]
    assert np.isclose(ccd_e, ref_e["ccd_e"])
    mycc = ccsd.CCSD(no)
    mycc.delta_e = 1e-11
    ccsd_e = mycc.solve(t_fock_pq, V_pqrs)["ccsd e"]
    assert np.isclose(ccsd_e, ref_e["ccsd_e"])


def main():
    test_ccsd_energy()


if __name__ == "__main__":
    main()
