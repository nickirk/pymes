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

        self.kCutoff = None
        
        self.gamma = None


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
    def init_single_basis(self, cutoff, nMax):
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
        imax = nMax #int(np.ceil(np.sqrt(cutoff*2)))
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


    def eval3BodyIntegrals(self, correlator = None,dtype=np.float64, sp=1):
        '''
            full 3-body integrals.
        '''
        algoName = "UEG.eval3BodyIntegrals"
        startTime = time.time()

        world = ctf.comm()
        rank = world.rank()

        if self.basis_fns == None:
            raise BasisSetNotInitialized(algoName)
        if correlator is None:
            self.correlator = self.trunc
            if rank == 0:
                print("\tNo correlator given.")
                print("\tUsing the default correlator: "+self.correlator.__name__)
        else:
            self.correlator = correlator
        if self.basis_indices_map is None:
            raise BasisFuncIndicesMapNotInitialised(algoName)

        if rank == 0:
            print(algoName)
            print("\tUsing TC method")
            print("\tUsing correlator:",correlator.__name__)
            print("\tkCutoff in correlator:",self.kCutoff)
    
        nP = int(len(self.basis_fns)/2)
        tV_opqrst = ctf.tensor([nP,nP,nP,nP,nP,nP], dtype=dtype, sp=sp)
        indices = []
        values = []

        # due to the momentum conservation, only 5 indices are free.
        # implementation follow closely the get_lmat_ueg in NECI
        numKInEachDir = self.imax*2+1

        for o in range(nP):
            if (o) % world.np() == rank:
                if rank == 0:
                    print("\tElapsed time={:.3f} s: calculating the {}-{} out of {} orbitals"\
                            .format(time.time()-startTime, o, \
                            o+world.np() if o+world.np() < nP else nP, nP))
            for r in range(nP):
                kIntVec1 = self.basis_fns[2*o].k - self.basis_fns[2*r].k
                for p in range(nP):
                    for s in range(nP):
                        kIntVec2 = self.basis_fns[2*p].k - self.basis_fns[2*s].k
                        for q in range(nP):
                            tIntVec = kIntVec1 + kIntVec2 + self.basis_fns[2*q].k
                            locT = numKInEachDir**2 * (tIntVec[0] + self.imax) + \
                                    numKInEachDir * (tIntVec[1] + self.imax) +\
                                    tIntVec[2] + self.imax
                            if locT < len(self.basis_indices_map) and locT >= 0:
                                t = int(self.basis_indices_map[locT])
                                if t < 0:
                                    continue
                            else:
                                continue

                            kVec1 = -2.0*np.pi/self.L*kIntVec1 
                            kVec2 = -2.0*np.pi/self.L*kIntVec2
                            kVec3 = -2.0*np.pi/self.L*(self.basis_fns[2*q].k-self.basis_fns[2*t].k) 


                            w12 = self.correlator(kVec1.dot(kVec1)) \
                                    * self.correlator(kVec2.dot(kVec2))\
                                    * kVec1.dot(kVec2)
                            w13 = self.correlator(kVec1.dot(kVec1))\
                                    * self.correlator(kVec3.dot(kVec3))\
                                    * kVec1.dot(kVec3)
                            w23 = self.correlator(kVec2.dot(kVec2))\
                                    * self.correlator(kVec3.dot(kVec3))\
                                    * kVec2.dot(kVec3)
                            w = (w12+w13+w23) / self.Omega**2
                            #w = (w12) / self.Omega**2
                            index = o*nP**5+p*nP**4+q*nP**3+r*nP**2+s*nP+t

                            values.append(w)
                            indices.append(index)
                            if index >= nP**6:
                                print("Index exceeds size of the tensor")

        tV_opqrst.write(indices,values)


        return tV_opqrst



    def eval2BodyIntegrals(self, correlator = None, rpaApprox= True, \
            only2Body=False, effective2Body= False, dtype=np.float64,sp=1):
        world = ctf.comm()
        algoName = "UEG.eval2BodyIntegrals"
        startTime = time.time()

        rank = world.rank()
        if rank == 0:
            print(algoName)

        if self.basis_fns == None:
            raise BasisSetNotInitialized(algoName)

        if correlator is not None:
            self.correlator = correlator
            if rank == 0:
                print("\tUsing TC method")
                print("\tUsing correlator:",correlator.__name__)
                print("\tkCutoff in correlator:",self.kCutoff)
                print("\tIncluding only 2-body terms:", only2Body)
                print("\tIncluding only RPA approximation for 3-body:",rpaApprox)
                print("\tIncluding approximate 2-body terms from 3-body:", effective2Body)
    
        nP = int(len(self.basis_fns)/2)
        tV_pqrs = ctf.tensor([nP,nP,nP,nP], dtype=dtype, sp=sp)
        indices = []
        values = []
        
    
        numKInEachDir = self.imax*2+1

        for p in range(nP):
            if (p) % world.np() == rank:
                if rank == 0:
                    print("\tElapsed time={:.3f} s: calculating the {}-{} out of {} orbitals"\
                            .format(time.time()-startTime, p, \
                            p+world.np() if p+world.np() < nP else nP, nP))
                for r in range(nP):
                    dIntK = self.basis_fns[p*2].k-self.basis_fns[r*2].k
                    dKVec = self.basis_fns[p*2].kp-self.basis_fns[r*2].kp
                    uMat  = 0.
                    if correlator is not None:
                        uMat = self.sumNablaUSquare(dKVec) 

                    for q in range(nP):
                        intKS = self.basis_fns[q*2].k + dIntK
                        #if self.is_k_in_basis(intKS):
                        locS = numKInEachDir**2 * (intKS[0] + self.imax) + \
                                numKInEachDir * (intKS[1] + self.imax) +\
                                intKS[2] + self.imax
                        if locS < len(self.basis_indices_map) and locS >= 0:
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
                        elif rpaApprox:
                            # tc
                            if np.abs(dkSquare) > 0.:
                                rs_dk = self.basis_fns[r*2].kp-self.basis_fns[s*2].kp
                                w = 4.*np.pi/dkSquare \
                                        +  uMat\
                                        + dkSquare * correlator(dkSquare)\
                                        - (rs_dk.dot(-dKVec)) * correlator(dkSquare) \
                                        - (self.nel-2)*dkSquare*correlator(dkSquare)**2/self.Omega
                                w = w / self.Omega
                            else:
                                #w = correlator(dkSquare, multKSquare=True) + uMat
                                w =  uMat / self.Omega
                                #w =  0.
                            indices.append(nP**3*p + nP**2*q + nP*r + s)
                            values.append(w)
                        elif only2Body:
                            if np.abs(dkSquare) > 0.:
                                rs_dk = self.basis_fns[r*2].kp-self.basis_fns[s*2].kp
                                w = 4.*np.pi/dkSquare \
                                        +  uMat\
                                        + dkSquare * correlator(dkSquare)\
                                        - (rs_dk.dot(-dKVec)) * correlator(dkSquare)
                                        #- (self.nel - 2)*dkSquare*correlator(dkSquare)**2/self.Omega
                                w = w / self.Omega
                            else:
                                #w = correlator(dkSquare, multKSquare=True) + uMat
                                w =  uMat / self.Omega
                                
                                # \sum_k' (k-k')k'u(k-k')u(k')

                            indices.append(nP**3*p + nP**2*q + nP*r + s)
                            values.append(w)
                        elif effective2Body:
                            if np.abs(dkSquare) > 0.:
                                rs_dk = self.basis_fns[r*2].kp-self.basis_fns[s*2].kp
                                w = 4.*np.pi/dkSquare \
                                        +  uMat\
                                        + dkSquare * correlator(dkSquare)\
                                        - (rs_dk.dot(-dKVec)) * correlator(dkSquare) \
                                        - (self.nel - 2)*dkSquare*correlator(dkSquare)**2/self.Omega\
                                        + 2.*self.contractExchange3Body(self.basis_fns[2*p].kp, dKVec)\
                                        - 2.*self.contractExchange3Body(self.basis_fns[2*p].kp - dKVec, dKVec)\
                                        + 2.*self.contractP_KWithQ(self.basis_fns[2*p].kp, dKVec)
                                w = w / self.Omega
                            else:
                                #w = correlator(dkSquare, multKSquare=True) + uMat
                                w =  uMat / self.Omega
                            indices.append(nP**3*p + nP**2*q + nP*r + s)
                            values.append(w)
                            #tV_pqrs[p,q,r,s] = 4.*np.pi/dkSquare/sys.Omega
        tV_pqrs.write(indices,values)
        return tV_pqrs


    def contractExchange3Body(self, pVec, kVec):
        '''
        pVec and kVec should be a single vector
        kVec is the momentum transfer
        '''
        pPrim = np.array([self.basis_fns[i*2].kp for i in range(int(self.nel/2))])
        pVec = pVec-pPrim
        kVecSquare = np.einsum("i,i->", kVec, kVec)
        pVecSquare = np.einsum("ni,ni->n", pVec, pVec)
        pVecDotKVec = np.einsum("ni,i->n", pVec, kVec)
        result = pVecDotKVec * self.correlator(kVecSquare) \
                * self.correlator(pVecSquare)
        result = np.einsum("n->", result) / self.Omega


        return result


    def contractP_KWithQ(self, pVec, kVec):

        pPrim = np.array([self.basis_fns[i*2].kp for i in range(int(self.nel/2))])
        vec1 = pVec - kVec - pPrim
        vec2 = pVec-pPrim

        dotProduct = np.einsum("ni,ni->n", vec1, vec2)
        vec1Square = np.einsum("ni,ni->n", vec1, vec1)
        vec2Square = np.einsum("ni,ni->n", vec2, vec2)
        result = dotProduct * self.correlator(vec1Square) * self.correlator(vec2Square)

        result = np.einsum("n->",result) / self.Omega

        return result


    def contract3BodyIntegralsTo2Body(self, integrals):
        # factor 2 for the spin degeneracy
        # factor (self.nel-2)/self.nel 
        fac = 2.*(self.nel-2.)/self.nel
        RPA2Body = fac*ctf.einsum("opqrsq->oprs", integrals)
        return RPA2Body

    def sumNablaUSquare(self, k, cutoff=30):
        # need to test convergence of this cutoff
        if self.kPrime is None:
            self.kPrime = np.array([[i,j,k] for i in range(-cutoff,cutoff+1) \
                    for j in range(-cutoff,cutoff+1) for k in range(-cutoff,cutoff+1)])
        k1 = 2*np.pi*self.kPrime/self.L
        k2 = k - k1

        k1Square = np.einsum("ni,ni->n", k1,k1)
        k2Square = np.einsum("ni,ni->n", k2,k2)
        k1DotK2 = np.einsum("ni,ni->n", k1,k2)
        result = k1DotK2 * self.correlator(k1Square) * self.correlator(k2Square)
        result = np.einsum("n->", result) / self.Omega

        return result



    def yukawa(self, kSquare, multKSquare=False):
        '''
        The G=0 terms need more consideration
        '''
        if self.gamma is None:
            rho = self.nel / self.Omega 
            gamma = np.sqrt(4.* np.pi * rho)
            #gamma = np.sqrt(4.*(3.*rho/np.pi)**(1/3.))
        else:
            gamma = self.gamma
        a = -4.*np.pi/gamma
        if self.kCutoff is not None:
            kCutoffSquare = self.kCutoff * ((2*np.pi/self.L)**2)
            kCutoffDenom = kCutoffSquare*(kCutoffSquare + gamma**2)
        else:
            kCutoffDenom = 1e-12
        if not multKSquare:
            b = kSquare*(kSquare+gamma**2)
            result = np.divide(a , b, out = np.zeros_like(b), where=np.abs(b)>kCutoffDenom)
        else:
            if kSquare > kCutoffSquare:
                result = a/(kSquare+gamma**2)
            else:
                result = 0.

        return result


    def trunc(self, kSquare, multKSquare=False):
        '''
        The G=0 terms need more consideration
        '''
        if self.kCutoff is None:
            self.kCutoff = int(ceil(np.sqrt(self.cutoff)))

        if self.gamma is None:
            self.gamma = 1.0
        
        kCutoffSquare = (self.kCutoff * 2*np.pi/self.L)**2
        #kCutoffSquare = self.kCutoff**2
        #if (kSquare <= ktc_cutoffSquare):
        #    result = 0.
        #else:
        #    result = - 12.566370614359173 / kSquare/kSquare
        
        if type(kSquare) is not np.ndarray:
            if kSquare <= kCutoffSquare:
                kSquare = 0.
        else:
            kSquare[kSquare <= kCutoffSquare] = 0.
        result = np.divide(-4.*np.pi, kSquare**2, out = np.zeros_like(kSquare),\
                where=kSquare>1e-12)
        return result*self.gamma

    def coulomb(self, kSquare, multKSquare=False):
        '''
        The G=0 terms need more consideration
        '''
        if self.gamma is None:
            gamma = 1.
        else:
            gamma = self.gamma
        result = np.divide(-4.*np.pi*gamma, kSquare, out = np.zeros_like(kSquare),\
                where=kSquare>1e-12)
        return result

    def stg(self,kSquare,multKSquare=False):
        if self.gamma is None:
            rho = self.nel / self.Omega 
            gamma = np.sqrt(4.* np.pi * rho)
            #gamma = np.sqrt(4.*(3.*rho/np.pi)**(1/3.))
        else:
            gamma = self.gamma
        a = -4.*np.pi/gamma
        if self.kCutoff is not None:
            kCutoffSquare = self.kCutoff * ((2*np.pi/self.L)**2)
            kCutoffDenom = (kCutoffSquare + gamma**2)**2
        else:
            kCutoffDenom = 1e-12
        if not multKSquare:
            b = (kSquare+gamma**2)**2
            result = np.divide(a , b, out = np.zeros_like(b), where=np.abs(b)>kCutoffDenom)

        return result



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
