# create a pyscf H2O molecule
from pyscf import gto, scf, ao2mo, cc
from pyscf.tools import fcidump 
import numpy as np
from functools import reduce

from pymes.solver import feast_eom_rccsd

def driver():
    mol = gto.Mole(
        atom = 'O	0.0000	0.0000	0.1173; H	0.0000	0.7572	-0.4692; H	0.0000	-0.7572	-0.4692',
        basis = 'aug-cc-pvtz',
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
    #e, _ = mycc.eomee_ccsd_singlet(nroots=20)
    #e, _ = mycc.eeccsd(nroots=28)
    #print(e)
    #np.save("eom_ccsd_pyscf_all.npy", e)

    # EOM-EE-CCSD calculation
    eom = feast_eom_rccsd.FEAST_EOMEESinglet(mycc)
    eom.max_cycle = 20
    eom.ls_max_iter = 10
    eom.conv_tol = 1e-6
    e_feast, _ = eom.kernel(nroots=5, e_c=19.74, e_r=0.1)
    np.save("eom_ccsd_feast.npy", e_feast)


def main():
    driver()

if __name__ == "__main__":
    main()