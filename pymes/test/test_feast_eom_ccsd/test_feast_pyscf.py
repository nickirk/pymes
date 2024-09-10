# create a pyscf H2O molecule
from pyscf import gto, scf, ao2mo, cc
from pyscf.lib import logger
import numpy as np
from functools import reduce

from pymes.solver import feast_eom_rccsd


def driver():
    basis = '6311g**'
    #basis = 'aug-ccpvtz'
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
    logger.verbose = 2
    eom.max_cycle = 20
    eom.ls_max_iter = 1
    eom.conv_tol = 1e-7
    eom.max_ntrial = 7
    eom.verbose = 5
    eom.nroots = 7
    # constructing guess
    nocc = mycc.nocc
    nvir = mycc.nmo - nocc
    #r1 = (np.random.random((nocc, nvir))-0.5)*0.0001
    #r2 = (np.random.random((nocc, nocc, nvir, nvir)) - 0.5)*0.00001
    r1 = np.zeros((nocc, nvir))
    r2 = np.zeros((nocc, nocc, nvir, nvir))
    r1[0, 0] = 1
    #r1 /= np.linalg.norm(r1)
    #r1[0, 5] = 
    guess = mycc.amplitudes_to_vector(r1, r2)
    print(" loc = ", np.argmax(np.abs(guess)))
    emin = 1.1441
    de = 3
    emax = emin + de*0.99999
    energies = []
    r1 = []
    r2 = []
    for emin_ in np.arange(emin, emax, de):
        print("emin = ", emin_, "emax = ", emin_+de)
    
        e_feast, u_vecs = eom.kernel(nroots=4,  emin=emin_, emax=emin_+de, ngl_pts=8, guess=[guess], e_guess_init=19.66)
        for u in u_vecs:
            r1_, r2_ = eom.vector_to_amplitudes(u)
            r1.append(r1_)
            r2.append(r2_)
        print("r1 norm: ", np.linalg.norm(np.asarray(r1), axis=(1,2)))
        np.save(f"eom_ccsd_feast_{emin_:.4f}_{(emin_+de):.4f}.{basis}.npy", e_feast)
        np.save(f"r2_feast_{emin_:.4f}_{(emin_+de):.4f}.{basis}.npy", np.asarray(r2))
        np.save(f"r1_feast_{emin_:.4f}_{(emin_+de):.4f}.{basis}.npy", np.asarray(r1))
        energies.append(e_feast)
    print("feast energies: ", energies)
    #print("valid targets: ", e[np.logical_and(e > emin, e < emax)])
    #print("all energies : ", e)

def main():
    driver()

if __name__ == "__main__":
    main()