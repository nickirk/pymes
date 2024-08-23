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