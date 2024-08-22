# create a pyscf H2O molecule
from pyscf import gto, scf, ao2mo, cc
from pyscf.tools import fcidump 
import numpy as np
from functools import reduce

from pymes.solver import feast_eom_rccsd

def driver():
    basis = 'sto6g'
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
    e, _ = mycc.eomee_ccsd_singlet(nroots=30)
    #e, _ = mycc.eeccsd(nroots=28)
    #print(e)
    #np.save("eom_ccsd_pyscf_all.npy", e)

    # EOM-EE-CCSD calculation
    eom = feast_eom_rccsd.FEAST_EOMEESinglet(mycc)
    eom.max_cycle = 25
    eom.ls_max_iter = 20
    eom.conv_tol = 1e-7
    eom.max_ntrial = 7
    e_c = 1.2
    e_r = 0.8
    e_feast, u_vecs = eom.kernel(nroots=1, e_c=e_c, e_r=e_r)
    np.save("eom_ccsd_feast_"+str(e_c)+"_"+str(e_r)+"."+basis+".npy", e_feast)
    np.save("u_vecs_feast_"+str(e_c)+"_"+str(e_r)+"."+basis+".npy", np.asarray(u_vecs))
    print("valid targets: ", e[np.logical_and(e > e_c - e_r, e < e_c + e_r)])

def main():
    driver()

if __name__ == "__main__":
    main()