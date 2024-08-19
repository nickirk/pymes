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
    mycc.max_memory = 12000
    mycc.incore_complete = True
    #mycc.eomee_ccsd_singlet(nroots=20)

    # EOM-EE-CCSD calculation
    eom = feast_eom_rccsd.FEAST_EOMEESinglet(mycc)
    eom.max_cycle = 50
    eom.ls_max_iter = 5
    eom.kernel(nroots=6, e_c=19.7, e_r=0.1)


def main():
    driver()

if __name__ == "__main__":
    main()