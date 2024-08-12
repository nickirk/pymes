import numpy as np

import ctf
from pymes.util import fcidump
from pymes.solver import ccsd, rt_eom_ccsd 
from pymes.mean_field import hf
from pymes.integral.partition import part_2_body_int

def driver(fcidump_file="pymes/test/test_eom_ccsd/FCIDUMP.LiH.321g", 
           ref_e={  "hf_e": -7.92958534362757, 
                    "ccsd_e": -0.0190883270951031,
                    "ee": [0.1180867117168979, 0.154376205595602]}):
    hf_ref_e = ref_e["hf_e"]
    n_elec, nb, e_core, e_orb, h_pq, V_pqrs = fcidump.read(fcidump_file)

    t_V_pqrs = ctf.astensor(V_pqrs)
    t_h_pq = ctf.astensor(h_pq)


    no = int(n_elec/2)
    # make sure HF energy is correct first
    hf_e = hf.calc_hf_e(no, e_core, t_h_pq, t_V_pqrs)
    print("HF e = ", hf_e)
    #assert np.isclose(hf_e, hf_ref_e)

    # CCSD energies
    t_fock_pq = hf.construct_hf_matrix(no, t_h_pq, t_V_pqrs)
    mycc = ccsd.CCSD(no)
    mycc.delta_e = 1e-12
    mycc.max_iter = 200
    ccsd_result = mycc.solve(t_fock_pq, t_V_pqrs, max_iter=200)
    ccsd_e = ccsd_result["ccsd e"]

    ccsd_e_ref = ref_e["ccsd_e"]
    #assert np.isclose(ccsd_e, ccsd_e_ref)

    # construct a EOM-CCSD instance
    # current formulation requires the singles dressed fock and V tensors
    # partition V integrals

    t_T_ai = ccsd_result["t1"].copy()
    t_T_abij = ccsd_result["t2"].copy()

    dict_t_V = part_2_body_int(no, t_V_pqrs)

    t_fock_dressed_pq = mycc.get_T1_dressed_fock(t_fock_pq, t_T_ai, dict_t_V)
    dict_t_V_dressed = mycc.get_T1_dressed_V(t_T_ai, dict_t_V)#, dict_t_V_dressed)

    n_e = 2
    nv = t_T_ai.shape[0]
    u_singles_0 = np.random.random([nv, no]) - 0.5
    u_doubles_0 = np.zeros([nv, nv, no, no])
    # calculate the norm 
    u_vec = np.concatenate((u_singles_0.flatten(), u_doubles_0.flatten()), axis=0)
    norm = np.linalg.norm(u_vec)
    u_singles_0 = u_singles_0/norm
    u_doubles_0 = u_doubles_0/norm

   
    eom_cc = rt_eom_ccsd.RT_EOM_CCSD(no, e_c=0.5, e_r=0.1, max_iter=100, tol=1e-8)
    eom_cc.linear_solver = "jacobi"
    nt = 200
    dt = 0.5
    c_t = np.zeros(nt-1, dtype=complex)
    t = np.arange(1,nt)*dt
    u_singles = u_singles_0.copy()
    u_doubles = u_doubles_0.copy()
    for n in range(0,nt-1):
        ut_singles, ut_doubles = eom_cc.solve(
            t_fock_dressed_pq, dict_t_V_dressed, t_T_abij, 
            dt=dt, u_singles=u_singles, u_doubles=u_doubles)
        # update the u_singles and u_doubles
        u_singles = ut_singles.copy()
        u_doubles = ut_doubles.copy()
        ct_ = np.tensordot(u_singles_0, ut_singles, axes=2)
        ct_ += np.tensordot(u_doubles_0, ut_doubles, axes=4)
        print("ct = ", ct_)
        c_t[n] = ct_
        np.save("ct.npy", np.column_stack((t,c_t)))

def test_rt_eom_ccsd_model_ham():
    no = 4
    nv = 4
    dt = 0.3
    nt = 2000
    e_c = 8
    e_r = 1
    eom_cc = rt_eom_ccsd.RT_EOM_CCSD(no, e_c=e_c, e_r=e_r, max_iter=100, tol=1e-8)
    #ham = eom_cc.construct_fake_non_sym_ham(nv, no)
    #np.save("ham.npy", ham)
    ham = np.load("ham.npy")
    e_target, v_target = np.linalg.eig(ham)
    print("Target eigenvalues = ", e_target)
    #np.save("e_target.npy", e_target)
    # generate initial guess
    np.random.seed(None)
    u_singles_0 = (np.random.random([nv, no])-0.5)*1
    u_doubles_0 = (np.random.random([nv, nv, no, no])-0.5)*5
    # form the u_vec
    u_vec = np.concatenate((u_singles_0.flatten(), u_doubles_0.flatten()), axis=0)
    # normalize the u_vec
    u_singles_0 = u_singles_0/np.linalg.norm(u_vec)
    u_doubles_0 = u_doubles_0/np.linalg.norm(u_vec)

    c_t = np.zeros(nt-1, dtype=complex)
    t = np.arange(1,nt)*dt
    u_singles = u_singles_0.copy()
    u_doubles = u_doubles_0.copy()
    for n in range(0,nt-1):
        ut_singles, ut_doubles = eom_cc.solve_test(ham, dt, u_singles=u_singles, u_doubles=u_doubles)
        # update the u_singles and u_doubles
        u_singles = ut_singles.copy()
        u_doubles = ut_doubles.copy()
        ct_ = np.tensordot(u_singles_0, ut_singles, axes=2)
        ct_ += np.tensordot(u_doubles_0, ut_doubles, axes=4)
        print("ct = ", ct_)
        c_t[n] = ct_
        np.save("ct1.npy", np.column_stack((t,c_t)))
    # print the eigenvalues of the target hamiltonian that are within the range of e_c-e_r and e_c+e_r
    e_target = np.sort(e_target)
    e_target = e_target[(e_target > e_c-e_r) & (e_target < e_c+e_r)]
    print("Eigenvalues within the range of e_c-e_r and e_c+e_r = ", e_target)

def signal_processing():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit
    
    # Assuming ct[:,0] contains the time data and ct[:,1] contains the signal data
    time_data = ct[:, 0]
    signal_data = ct[:, 1]
    
    # Compute the sampling rate (assuming uniform sampling)
    delta_t = time_data[1] - time_data[0]
    
    # FFT of the signal
    ct_fft = np.fft.fft(signal_data)
    ct_fft = ct_fft / len(ct_fft)  # Normalize the FFT
    ct_fft = ct_fft * 2
    ct_fft[0] = ct_fft[0]
    #ct_fft = np.fft.fftshift(ct_fft)
    
    # Compute the frequency array and convert to angular frequency
    ct_freq = np.fft.fftfreq(len(ct_fft), d=delta_t)
    #positive_freq_indices = ct_freq >= 0
    #ct_freq = ct_freq[positive_freq_indices]
    #ct_fft = ct_fft[positive_freq_indices]
    
    ct_omega =  (2 * np.pi * ct_freq)
    ct_omega = (np.fft.fftshift(ct_omega)).real
    # Identify the peaks
    peaks, _ = find_peaks(np.abs(ct_fft), height=0.005)
    print(peaks)
    print(np.real(ct_omega[peaks]))
    
    
    # Plot the FFT with the correct x-axis
    plt.figure()
    #plt.plot(ct_omega, np.abs(ct_fft.real), '--', label='FFT real')
    #plt.plot(ct_omega, np.abs(ct_fft.imag), '--', label='FFT imag')
    plt.plot(ct_omega, np.abs(ct_fft), label='FFT magnitude')
    # Plot the fitted composite Lorentzian function
    # Define the Lorentzian function
    def composite_lorentzian(x, *params):
        n_peaks = len(params) // 3
        y = np.zeros_like(x)
        for i in range(n_peaks):
            A = params[3 * i]
            x0 = params[3 * i + 1]
            gamma = params[3 * i + 2]
            y += (A / np.pi) * (gamma / ((x - x0)**2 + gamma**2))
        return y
    
    # Initial guess for the parameters [A1, x01, gamma1, A2, x02, gamma2, ...]
    #initial_guess = []
    #for peak in peaks:
    #    initial_guess.extend([np.real(ct_fft)[peak], np.abs((ct_omega[peak])), 0.01])
    #
    ### Fit the composite Lorentzian function
    #popt, _ = curve_fit(composite_lorentzian, np.real(ct_omega), np.abs((ct_fft)), p0=initial_guess)
    #fitted_peak_locations = popt[1::3]
    ## Place a vertical line at the fitted peak locations
    #plt.vlines(fitted_peak_locations, 0, np.max(np.abs(ct_fft)), color='g', linestyle='--', label='Fitted Peaks')
    #print("Fitted peaks = ", popt[popt>1])
    #plt.plot(ct_omega, composite_lorentzian(ct_omega, *popt), label='Composite Lorentzian fit')
    
    plt.xlim(1, 3)
    plt.ylim(0, 0.03)
    plt.xlabel('Angular Frequency (rad/s)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    #test_rt_eom_ccsd_model_ham()
    driver()