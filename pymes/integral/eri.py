class ERI:

    def __init__(self, model=None):
        self.model = model
        self.n_elec = self.model.n_elec if model else None
        self.n_orb = len(self.model.basis_fns)//2 if model else None
        self.n_occ = self.n_elec // 2
        self.fock = None
        self.oooo = None
        self.ovvo = None
        self.voov = None 
        self.oovv = None
        self.ovov = None
        self.vvoo = None
        self.vvvv = None
    
    def calc_eri(self, incore=True):
        """
        Calculate the electron repulsion integrals (ERIs) for the system.
        
        Parameters:
        incore (bool): If True, store ERIs in memory. If False, calculate on-the-fly.
        
        Returns:
        eri (numpy.ndarray): The calculated ERIs.
        """
        if incore:
            self.fock = self.model.fock()
            V_pqrs = self.model.get_g2b()
            self.part_eri(self.fock, V_pqrs)
        else:
            # Placeholder for on-the-fly calculation
            self.oooo = self.model.get_g2b('oooo')
            self.ovvo=  self.model.get_g2b('ovvo')
            self.voov=  self.model.get_g2b('voov')
            self.oovv=  self.model.get_g2b('oovv')
            self.ovov=  self.model.get_g2b('ovov')
            self.vvoo=  self.model.get_g2b('vvoo')

    def part_eri(self, fock, V_pqrs):    
        no = self.n_occ
        self.fock = fock
        self.oooo= V_pqrs[:no, no:, no:, :no]
        self.ovvo= V_pqrs[:no, no:, no:, :no]
        self.voov= V_pqrs[no:, :no, :no, :no]
        self.oovv= V_pqrs[:no, :no, no:, no:]
        self.ovov= V_pqrs[:no, no:, :no, no:]
        self.vvoo= V_pqrs[no:, no:, :no, :no]
        self.vvvv= V_pqrs[no:, no:, no:, no:]
    
    def get_vvvv(self, idx=None):
        """
        Get the electron repulsion integrals for the virtual-virtual-virtual-virtual (vvvv) block.
        
        Parameters:
        idx (tuple): A tuple specifying the range of indices to extract.
            idx[0] (int): The start of 0th index 
            idx[1] (int): The end of 0th index
            idx[2] (int): The start of 1st index
            idx[3] (int): The end of 1st index
            idx[4] (int): The start of 2nd index
            idx[5] (int): The end of 2nd index
            idx[6] (int): The start of 3rd index
            idx[7] (int): The end of 3rd index

        
        Returns:
        vvvv (numpy.ndarray): The extracted vvvv block of ERIs.
        """
        if self.vvvv is not None:
            if idx is None:
                return self.vvvv
            else:
                return self.vvvv[idx[0]:idx[1], idx[2]:idx[3], idx[4]:idx[5], idx[6]:idx[7]]
        else:
            raise ValueError("VVVV block is not calculated or available.")