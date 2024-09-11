# create a pyscf H2O molecule
from pyscf import gto, scf, ao2mo, cc
from pyscf.lib import logger
import numpy as np
from functools import reduce

from pymes.solver import feast_eom_rccsd


def driver():
    basis = '6311g**'
    #basis = 'aug-ccpvtz'
    #basis = 'sto6g'
    mol = gto.Mole(
        atom = 'O 0.0000	0.0000	0.1185; H 0.0000	0.7555	-0.4739; H 0.0000 -0.7555 -0.4739',
        basis = basis,
        verbose = 4,
        symmetry = True,
        unit = 'A'
    )
    mol.build()

    # RHF calculation
    mf = scf.RHF(mol)
    mf.verbose = 5
    mf.kernel()
    mf.analyze()

    # RCCSD calculation
    mycc = cc.CCSD(mf)
    mycc.kernel()
    #mycc.max_memory = 12000
    mycc.incore_complete = True
    #e, _ = mycc.eomee_ccsd_singlet(nroots=40)
    #np.save("eom_ccsd_pyscf_all.npy", e)
    ##e, _ = mycc.eeccsd(nroots=28)
    ##print(e)
    #e = np.load("eom_ccsd_pyscf_all.npy")

    # EOM-EE-CCSD calculation
    eom = feast_eom_rccsd.FEAST_EOMEESinglet(mycc)
    eom.max_cycle = 20
    eom.ls_max_iter = 5
    eom.conv_tol = 1e-7
    eom.max_ntrial = 7
    #eom.verbose = 5
    # constructing guess
    #nocc = mycc.nocc
    #nvir = mycc.nmo - nocc
    #r1 = np.zeros((nocc, nvir))
    #r2 = np.zeros((nocc, nocc, nvir, nvir))
    #r1[0, 0] = 1
    #r1 /= np.linalg.norm(r1)
    #r1[0, 5] = 
    #guess = mycc.amplitudes_to_vector(r1, r2)
    #print(" loc = ", np.argmax(np.abs(guess)))
    de = 3
    e_feast, _ = eom.kernel(nroots=3, ngl_pts=8, e_c=1.40, e_r=de, e_brd=1., n_aux=1)
    print("feast energies: ", e_feast)
    assert np.isclose(e_feast[0].real, 19.68806362) 

def main():
    driver()

if __name__ == "__main__":
    main()