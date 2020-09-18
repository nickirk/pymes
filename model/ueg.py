import time

import numpy as np
from pymes.basis_set import planewave
import ctf

class UEG:

    def __init__(self, nel, nalpha, nbeta, rs):
        #: number of electrons
        self.nel = nel
        #: number of alpha (spin-up) electrons
        self.nalpha = nalpha
        #: number of beta (spin-down) electrons
        self.nbeta = nbeta
        #: electronic density
        self.rs = rs
        #: length of the cubic simulation cell containing nel electrons
        #: at the density of rs
        self.L = self.rs*((4*np.pi*self.nel)/3)**(1.0/3.0)
        #: volume of the cubic simulation cell containing nel electrons at the density
        #: of rs
        self.Omega = self.L**3

        self.basis_fns = None

        self.imax = 0

        self.cutoff = 0.

        self.basis_indices_map = None
        
        self.kPrime = None

        self.correlator = None


    def is_k_in_basis(self,k):
        if k.dot(k) <= self.cutoff:
            return True
        return False

    def init_basis_indices_map(self):
        numKInEachDir = self.imax*2+1
        self.basis_indices_map = -1 * np.ones(numKInEachDir**3).astype(int)
        for i in range(int(len(self.basis_fns)/2)):
            s = numKInEachDir**2 * (self.basis_fns[i*2].k[0] + self.imax) + \
                    numKInEachDir * (self.basis_fns[i*2].k[1] + self.imax) +\
                    self.basis_fns[i*2].k[2] + self.imax
            self.basis_indices_map[s] = i
            

    #--- Basis set ---
    def init_single_basis(self, cutoff):
        '''Create single-particle basis.
    
    :type self: :class:`UEG`
    :param sys: UEG system to be studied.
    
    :param nMax: the number of G vectors in each direction to loop over. eg. 1 gives -1, 0, 1
    
    :param float cutoff: energy cutoff, in units of `2*(2\pi/L)^2`, defining the single-particle basis.  Only single-particle basis functions with a kinetic energy equal to or less than the cutoff are considered.
    
    :type sym: np.array
    :param sym: integer vector defining the wavevector, in units of `2\pi/L`, representing the desired symmetry.  Only Hartree products and determinants of this symmetry are returned.
    
    :returns: basis_fns where:
    
        basis_fns
            tuple of relevant :class:`PlaneWaveFn` objects, ie the single-particle basis set.
    
    '''
    
        # Single particle basis within the desired energy cutoff.
        #cutoff = cutoff*(2*np.pi/self.L)**2
        imax = int(np.ceil(np.sqrt(cutoff*2)))
        self.cutoff = cutoff
        self.imax = imax
        basis_fns = []
        for i in range(-imax, imax+1):
            for j in range(-imax, imax+1):
                for k in range(-imax, imax+1):
                    bfn = planewave.BasisFunc(i, j, k, self.L, 1)
                    if self.is_k_in_basis(bfn.k):
                        basis_fns.append(planewave.BasisFunc(i, j, k, self.L, 1))
                        basis_fns.append(planewave.BasisFunc(i, j, k, self.L, -1))
        # Sort in ascending order of kinetic energy.  Note that python's .sort()
        # (since 2.3) is guaranteed to be stable.
        basis_fns.sort()
        basis_fns = tuple(basis_fns)
        self.basis_fns = basis_fns

        self.init_basis_indices_map()
    
        return basis_fns

    def eval2BodyIntegrals(self, correlator = None, dtype=np.float64,sp=1):
        world = ctf.comm()
        algoName = "UEG.evalCoulIntegrals"

        if self.basis_fns == None:
            raise BasisSetNotInitialized(algoName)

        if correlator is not None:
            self.correlator = correlator
    
        nP = int(len(self.basis_fns)/2)
        tV_pqrs = ctf.tensor([nP,nP,nP,nP], dtype=dtype, sp=sp)
        indices = []
        values = []
        
        #basisKp = [ self.basis_fns[2*i].kp.tolist() for i in range(nP)]
        
    
        rank = world.rank()
        numKInEachDir = self.imax*2+1
        if rank == 0:
            print(algoName)

        startTime = time.time()
        for p in range(nP):
            if (p) % world.np() == rank:
                if rank == 0:
                    print("\tElapsed time=%d s: %i in %i finished" % (time.time()-startTime, p, nP))
                for r in range(nP):
                    dIntK = self.basis_fns[p*2].k-self.basis_fns[r*2].k
                    dKVec = self.basis_fns[p*2].kp-self.basis_fns[r*2].kp
                    uMat  = 0.
                    if correlator is not None:
                        uMat = self.sumNablaUSquare(dKVec) 
                        print("dK=", dIntK, uMat)

                    for q in range(nP):
                        intKS = self.basis_fns[q*2].k + dIntK
                        #if self.is_k_in_basis(intKS):
                        locS = numKInEachDir**2 * (intKS[0] + self.imax) + \
                                numKInEachDir * (intKS[1] + self.imax) +\
                                intKS[2] + self.imax
                        if locS < len(self.basis_indices_map):
                            s = int(self.basis_indices_map[locS])
                            if s < 0:
                                continue
                        else:
                            continue
                        dkSquare = dKVec.dot(dKVec)
                        if correlator is None:
                            if np.abs(dkSquare) > 0.:
                                w = 4.*np.pi/dkSquare/self.Omega
                                indices.append(nP**3*p + nP**2*q + nP*r + s)
                                values.append(w)
                        else:
                            # tc
                            if np.abs(dkSquare) > 0.:
                                rs_dk = self.basis_fns[r*2].kp-self.basis_fns[s*2].kp
                                w = 4.*np.pi/dkSquare \
                                        + dkSquare * correlator(dkSquare)\
                                        - (rs_dk.dot(-dKVec)) * correlator(dkSquare) \
                                        +  uMat
                                w = w / self.Omega
                            else:
                                w = correlator(dkSquare, multKSquare=True)- uMat
                                w = w / self.Omega
                                # \sum_k' (k-k')k'u(k-k')u(k')

                            indices.append(nP**3*p + nP**2*q + nP*r + s)
                            values.append(w)
                            #tV_pqrs[p,q,r,s] = 4.*np.pi/dkSquare/sys.Omega
        tV_pqrs.write(indices,values)
        return tV_pqrs

    def sumNablaUSquare(self, k, cutoff=20):
        # need to test convergence of this cutoff
        if self.kPrime is None:
            self.kPrime = np.array([[i,j,k] for i in range(-cutoff,cutoff+1) \
                    for j in range(-cutoff,cutoff+1) for k in range(-cutoff,cutoff+1)])
        k1 = 2*np.pi*self.kPrime/self.L
        k2 = k - k1

        k1Square = np.einsum("ni,ni->n", k1,k1)
        k2Square = np.einsum("ni,ni->n", k2,k2)
        k1DotK2 = np.einsum("ni,ni->n", k1,k2)
        #print(self.correlator(k1Square))
        #print(self.correlator(k2Square))
        result = k1DotK2 * self.correlator(k1Square) * self.correlator(k2Square)
        result = np.einsum("n->", result) / self.Omega

        return result

    #def calcNablaUMat(self):
    #    nablaUMat = np.zeros((2*self.imax))





    def yukawa(self, kSquare, multKSquare=False):
        '''
        The G=0 terms need more consideration
        '''
        rho = self.nel / self.Omega 
        muSquare = np.sqrt(4.*np.pi*rho)
        a = -4.*np.pi
        if not multKSquare:
            b = kSquare*(kSquare+muSquare*2)
            result = np.divide(a , b, out = np.zeros_like(b), where=np.abs(b)>1e-12)
        else:
            result = a/(kSquare+muSquare*2)

        return result



    def evalTransCorr2BodyIntegrals(correlator):
    
        world = ctf.comm()
        algoName = "UEG.evalTransCorr2BodyIntegrals"

        if self.basis_fns == None:
            raise BasisSetNotInitialized(algoName)
    
        nP = int(len(self.basis_fns)/2)
        tTC_pqrs = ctf.tensor([nP,nP,nP,nP], dtype=dtype, sp=sp)
        indices = []
        values = []
        
        #basisKp = [ self.basis_fns[2*i].kp.tolist() for i in range(nP)]
        
    
        rank = world.rank()
        numKInEachDir = self.imax*2+1
        if rank == 0:
            print(algoName)

        startTime = time.time()
        for p in range(nP):
            if p % world.np() == rank:
                if rank == 0:
                    print("\tElapsed time=%d s: %i in %i finished" % (time.time()-startTime, p, nP))
                for r in range(nP):
                    dIntK = self.basis_fns[p*2].k-self.basis_fns[r*2].k
                    dKVec = self.basis_fns[p*2].kp-self.basis_fns[r*2].kp
                    for q in range(nP):
                        intKS = self.basis_fns[q*2].k + dIntK
                        #if self.is_k_in_basis(intKS):
                        locS = numKInEachDir**2 * (intKS[0] + self.imax) + \
                                numKInEachDir * (intKS[1] + self.imax) +\
                                intKS[2] + self.imax
                        if locS < len(self.basis_indices_map):
                            s = int(self.basis_indices_map[locS])
                            if s < 0:
                                continue
                        else:
                            continue
                        dkSquare = dKVec.dot(dKVec)
                        if np.abs(dkSquare) > 0.:
                            indices.append(nP**3*p + nP**2*q + nP*r + s)
                            values.append(4.*np.pi/dkSquare/self.Omega)
                            #tV_pqrs[p,q,r,s] = 4.*np.pi/dkSquare/sys.Omega
        tV_pqrs.write(indices,values)
    
        return tTC_pqrs
    
    def evalTransCorrThreeBodyIntegrals(tV_pqrs, basis_fns, correlator):
    
    
        return tTC_opqrst

    def calcGamma(self, overlap_basis, nP):
        '''
        FTOD : Fourier Transformed Overlap Density
        C^p_q({\bf G}) = \int\mathrm d{\bf r} \phi^*_p({\bf r}\phi_q({\bf r})e^{i{\bf G\cdot r}} 
        '''
        algoName = "UEG.calcGamma"
        if self.basis_fns == None:
            raise BasisSetNotInitialized(algoName)

        nG = int(len(overlap_basis)/2)
        gamma_pqG = np.zeros((nP,nP,nG))
    
        for p in range(0,nP,1):
            for q in range(0,nP,1):
                for g in range(0,nG,1):
                    if ((basis[2*p].k-basis[2*q].k) == overlap_basis[2*g].k).all():
                        GSquare = overlap_basis[2*g].kp.dot(overlap_basis[2*g].kp)
                        if np.abs(GSquare) > 1e-12 :
                            gamma_pqG[p,q,g] = np.sqrt(4.*np.pi/GSquare/self.Omega)
        return gamma_pqG
