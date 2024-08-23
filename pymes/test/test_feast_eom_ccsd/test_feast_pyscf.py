# create a pyscf H2O molecule
from pyscf import gto, scf, ao2mo, cc
from pyscf.lib import logger
import numpy as np
from functools import reduce

from pymes.solver import feast_eom_rccsd


def driver():
    basis = 'aug-ccpvtz'
    mol = gto.Mole(
        atom = 'O	0.0000	0.0000	0.1173; H	0.0000	0.7572	-0.4692; H	0.0000	-0.7572	-0.4692',
        basis = basis,
        verbose = 4,
        unit = 'A'
    )
    mol.build()

    # RHF calculation
    mf = scf.RHF(mol)
    mf.kernel()

    # RCCSD calculation
    mycc = cc.CCSD(mf)
    mycc.kernel()
    #mycc.max_memory = 12000
    mycc.incore_complete = True
    #e, _ = mycc.eomee_ccsd_singlet(nroots=30)
    #e, _ = mycc.eeccsd(nroots=28)
    #print(e)
    #np.save("eom_ccsd_pyscf_all.npy", e)

    # EOM-EE-CCSD calculation
    eom = feast_eom_rccsd.FEAST_EOMEESinglet(mycc)
    logger.verbose = 5 
    eom.max_cycle = 25
    eom.ls_max_iter = 20
    eom.conv_tol = 1e-7
    eom.max_ntrial = 7

    emin = 19.66
    emax = 19.68
    de = 0.01
    energies = []
    for emin_ in np.arange(emin, emax, de):
        print("emin = ", emin_, "emax = ", emin_+de)
    
        e_feast, u_vecs = eom.kernel(nroots=3,  emin=emin_, emax=emin_+de)
        np.save("eom_ccsd_feast_{emin_:.4f}_{emin_+de:.4f}.{basis}.npy", e_feast)
        np.save("u_vecs_feast_{emin_:.4f}_{emin_+de:.4f}.{basis}.npy", np.asarray(u_vecs))
        energies.append(e_feast)
    print("energies: ", energies)
    #print("valid targets: ", e[np.logical_and(e > emin, e < emax)])

def main():
    driver()

if __name__ == "__main__":
    main()