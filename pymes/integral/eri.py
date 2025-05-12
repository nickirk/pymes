from pymes.util.tensors import get_block_index

class ERI:

    def __init__(self, model=None):
        self.model = model
        self.n_elec = self.model.n_ele if model else None
        self.n_orb = len(self.model.basis_fns)//2 if model else None
        self.n_occ = self.n_elec // 2
        self.EHF   = None
        self.eps_occ  = None
        self.eps_virt = None
        self.fock  = None
        self.oooo  = None
        self.ovvo  = None
        self.voov  = None 
        self.oovv  = None
        self.vvoo  = None
        self.ovov  = None
        self.vovo  = None
        self.vvvv  = None
    
    def calc_eri(self, incore=True):
        """
        Calculate the electron repulsion integrals (ERIs) for the system.
        
        Parameters:
        incore (bool): If True, store ERIs in memory. If False, calculate on-the-fly.
        
        Returns:
        eri (numpy.ndarray): The calculated ERIs.
        """

        nP = self.n_orb
        no = self.n_occ

        if incore:

            self.EHF, self.eps_occ, self.eps_virt, self.fock, \
                self.oooo, self.vovo, self.voov = self.model.get_fock()

            idx    = get_block_index( 'full', nP, no)
            V_pqrs = self.model.get_2b_int( idx )

            self.part_eri(self.fock, V_pqrs)

        else:
            self.EHF, self.eps_occ, self.eps_virt, self.fock, \
                self.oooo, self.vovo, self.voov = self.model.get_fock()
            idx    = get_block_index( 'ovvo', nP, no)
            self.ovvo =  self.model.get_2b_int( idx )
            idx    = get_block_index( 'oovv', nP, no)
            self.oovv =  self.model.get_2b_int( idx )
            idx    = get_block_index( 'ovov', nP, no)
            self.ovov =  self.model.get_2b_int( idx)
            idx    = get_block_index( 'vvoo', nP, no)
            self.vvoo =  self.model.get_2b_int( idx )

    def part_eri(self, fock, V_pqrs):    
        no = self.n_occ
        self.fock = fock
        self.oooo = V_pqrs[:no, :no, :no, :no]
        self.ovvo = V_pqrs[:no, no:, no:, :no]
        self.voov = V_pqrs[no:, :no, :no, no:]
        self.oovv = V_pqrs[:no, :no, no:, no:]
        self.vvoo = V_pqrs[no:, no:, :no, :no]
        self.ovov = V_pqrs[:no, no:, :no, no:]
        self.vovo = V_pqrs[no:, :no, no:, :no]
        self.vvvv = V_pqrs[no:, no:, no:, no:]
    
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
            if idx is None:
                nP  = self.n_orb
                no  = self.n_occ
                idx = get_block_index('vvvv', nP, no)
                return self.model.get_2b_int( idx )
            else:
                no  = self.n_occ
                global_idx = tuple((no+idx[0], no+idx[1], \
                                    no+idx[2], no+idx[3], \
                                    no+idx[4], no+idx[5], \
                                    no+idx[6], no+idx[7]))
                return self.model.get_2b_int( global_idx )