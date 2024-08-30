# create a pyscf H2O molecule
from pyscf import gto, scf, ao2mo, cc
from pyscf.lib import logger
import numpy as np
from functools import reduce

from pymes.solver import feast_eom_rccsd


def driver():
    #basis = '6311g**'
    basis = 'ccpvtz'
    mol = gto.Mole(
        atom = 'O 0.0000	0.0000	0.1185; H 0.0000	0.7555	-0.4739; H 0.0000 -0.7555 -0.4739',
        basis = basis,
        verbose = 4,
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
    #e, _ = mycc.eomee_ccsd_singlet(nroots=28)
    ##e, _ = mycc.eeccsd(nroots=28)
    ##print(e)
    #e = np.load("eom_ccsd_pyscf_all.npy")

    # EOM-EE-CCSD calculation
    eom = feast_eom_rccsd.FEAST_EOMEESinglet(mycc)
    logger.verbose = 5 
    eom.max_cycle = 100
    eom.ls_max_iter = 3
    eom.conv_tol = 1e-7
    eom.max_ntrial = 7
    #eom.verbose = 5
    eom.nroots = 7

    emin = 1.15
    emax = 1.19999
    de = 0.05
    energies = []
    r1 = []
    r2 = []
    for emin_ in np.arange(emin, emax, de):
        print("emin = ", emin_, "emax = ", emin_+de)
    
        e_feast, u_vecs = eom.kernel(nroots=4,  emin=emin_, emax=emin_+de, ngl_pts=8)
        for u in u_vecs:
            r1_, r2_ = eom.vector_to_amplitudes(u)
            r1.append(r1_)
            r2.append(r2_)
        np.save(f"eom_ccsd_feast_{emin_:.4f}_{(emin_+de):.4f}.{basis}.npy", e_feast)
        np.save(f"r2_feast_{emin_:.4f}_{(emin_+de):.4f}.{basis}.npy", np.asarray(r2))
        np.save(f"r1_feast_{emin_:.4f}_{(emin_+de):.4f}.{basis}.npy", np.asarray(r1))
        energies.append(e_feast)
    print("energies: ", energies)
    #print("valid targets: ", e[np.logical_and(e > emin, e < emax)])

def main():
    driver()

if __name__ == "__main__":
    main()