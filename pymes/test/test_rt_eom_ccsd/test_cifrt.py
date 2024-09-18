import numpy as np

from pyscf import gto, scf, cc
from pymes.util import fcidump
from pymes.solver import rt_eom_rccsd 
from pymes.integral.partition import part_2_body_int

def driver():
    mol = gto.Mole(
        atom = 'O	0.0000	0.0000	0.1173; H	0.0000	0.7572	-0.4692; H	0.0000	-0.7572	-0.4692',
        basis = 'sto6g',
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
    e, _ = mycc.eomee_ccsd_singlet(nroots=20)
    #e, _ = mycc.eeccsd(nroots=28)
    #print(e)
    np.save("eom_ccsd_pyscf.npy", e)

    # EOM-EE-CCSD calculation
    eom = rt_eom_rccsd.CIFRT_EOMEESinglet(mycc)
    eom.max_cycle = 20
    eom.ls_max_iter = 10
    eom.conv_tol = 1e-6

    u_vec = np.random.random(eom.vector_size())-0.5
    u_vec = u_vec/np.linalg.norm(u_vec)


   
    nt = 2000
    dt = 0.5
    c_t = np.zeros(nt-1, dtype=complex)
    t = np.arange(1,nt)*dt
    u0 = u_vec.copy()
    for n in range(0,nt-1):
        u_vec_t = eom.kernel(guess=[u_vec], e_c=1.2, e_r=0.1, dt=dt)
        # update the u_singles and u_doubles
        ct_ = np.dot(u0, u_vec)
        u_vec = u_vec_t.copy()
        print("ct = ", ct_)
        c_t[n] = ct_
        np.save("ct.npy", np.column_stack((t,c_t)))
    np.save("u_vec_rt.npy", u_vec)


if __name__ == "__main__":
    #test_rt_eom_ccsd_model_ham()
    driver()